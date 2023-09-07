import logging
import os
import re
from typing import List
from urllib.parse import unquote

import openai
from overrides import override
from sqlalchemy import create_engine

from dataherald.db import DB
from dataherald.sql_database.models.types import DatabaseConnection
from dataherald.sql_generator import SQLGenerator
from dataherald.sql_generator.database_content_creator import FineTuningDatabaseContentCreator
from dataherald.sql_generator.generates_nl_answer import GeneratesNlAnswer
from dataherald.types import NLQuery, NLQueryResponse
from dataherald.utils.encrypt import FernetEncrypt

logger = logging.getLogger(__name__)


class FineTunedGPT(SQLGenerator):
    def output_parser(self, model_output: str) -> str:
        pattern = r'The SQL query I\'ll be generating is:(.*?)$'
        match = re.search(pattern, model_output, re.DOTALL)
        if match:
            sql = match.group(1).strip()
        else:
            sql = model_output
        re_combine_whitespace = re.compile(r"\s+")
        return re_combine_whitespace.sub(" ", sql).strip()


    @override
    def generate_response(
        self,
        user_question: NLQuery,
        database_connection: DatabaseConnection,
        context: List[dict] = None,
    ) -> NLQueryResponse:
        fernet_encrypt = FernetEncrypt()
        engine = create_engine(unquote(fernet_encrypt.decrypt(database_connection.uri)))
        db = FineTuningDatabaseContentCreator(engine)
        instruction = f"""
You are an assistant that is an expert in generating {db.dialect} SQL queries.
Having the access to database content, generate a correct {db.dialect} SQL query for the given question.
### Database content ###
    """
        database_content = db.get_table_info()
        system_content = instruction + database_content
        if context is not None:
            samples_prompt_string = "The following are some similar previous questions and their correct SQL queries from the database: \
            \n"
            for sample in context:
                samples_prompt_string += (
                    f"Question: {sample['nl_question']} \nSQL: {sample['sql_query']} \n"
                )

        question_with_context = (
            f"{user_question.question} An example of a similar question and the query that was generated \
                                to answer it is the following {samples_prompt_string}"
            if context is not None
            else user_question.question
        )
        response = None
        while response is None:
            try:
                response = openai.ChatCompletion.create(
                model="ft:gpt-3.5-turbo-0613:dataherald:spider:7t2q6Qhd",
                api_key=os.environ.get("OPENAI_API_KEY"),
                temperature=0.0,
                messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": question_with_context}
                    ]
                )
            except Exception as e:
                print(e)
                continue
        model_response = response.choices[0]['message']['content']
        sql = self.output_parser(model_response)
        nl_query_response = NLQueryResponse(
            nl_question_id=user_question.id,
            sql_query=sql,
        )
        generates_nl_answer = GeneratesNlAnswer(self.system, self.system.instance(DB))
        return generates_nl_answer.execute(nl_query_response)
