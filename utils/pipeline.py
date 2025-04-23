import json

from langchain_core.output_parsers import SimpleJsonOutputParser
from langchain.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI

import os


def model_dryrun(model):
    print('Model ' + model + ' loading!')
    question = "Fühlen Sie sich heute in der Lage diese Klausur zu schreiben?"
    options = {"A": "Ja", "B": "Nein, auf gar keinen Fall", "C": "Bringen wir es hinter uns!"}

    prompt_template = get_prompt_template()
    run_llm(
        prompt_template,
        prompt_variables_list=[
            {
                "question": question,
                "options": json.dumps(options),
                "begin_tokens": "[INST]",
                "end_tokens": "[/INST]",
            }
        ],
        model=model,
    )
    print()
    print('Model ' + model + ' loaded!')
    print()
    return True


def run_question(language, llm_model, question, options, is_openai=False):
    prompt_template = get_prompt_template(language)

    llm_outputs = run_llm(
        prompt_template,
        prompt_variables_list=[
            {
                "question": question,
                "options": json.dumps(options),
                "begin_tokens": "[INST]",
                "end_tokens": "[/INST]",
            }
        ],
        model=llm_model,
        is_openai=is_openai,
    )

    res = dict()
    res['text'] = question
    res['outcomes'] = llm_outputs
    return res


def run_question_with_rag(language, llm_model, context, question, options, is_openai=False):
    prompt_template = get_prompt_rag_template(language)

    llm_outputs = run_llm(
        prompt_template,
        prompt_variables_list=[
            {
                "context": " ".join(obj.text for obj in context),
                "question": question,
                "options": json.dumps(options),
                "begin_tokens": "[INST]",
                "end_tokens": "[/INST]",
            }
        ],
        model=llm_model,
        is_openai=is_openai,
    )

    res = dict()
    res['text'] = question
    res['outcomes'] = llm_outputs
    return res


def run_llm(prompt_template: PromptTemplate, prompt_variables_list: list[dict], model: str | None = None,
            is_openai=False):
    if is_openai:
        openai_api_key = os.getenv('OPENAI_API_KEY')
        llm = ChatOpenAI(model=model, temperature=0.1, openai_api_key=openai_api_key)

    else:
        ollama_host = os.getenv('OLLAMA_HOST')
        llm = OllamaLLM(format="json", temperature=0.1, base_url=ollama_host, model=model)

    llm_chain = prompt_template | llm | SimpleJsonOutputParser()

    responses = []
    for prompt_variables in prompt_variables_list:
        resp = llm_chain.invoke(prompt_variables)
        responses.append(resp)

    return responses


def get_prompt_rag_text(language):
    match language:
        case "en":
            return """
                    {begin_tokens}
                    Imagine you are an experienced doctor and need to answer a medical question.  
                    The answer choices are provided.  

                    ** Context: ** {context}
            
                    **Question:** {question}  
                    **Answer choices:** {options}  

                    ### Important Instructions:
                    - Your response **must** be a **valid JSON output**.  
                    - **No additional text, explanations, or formatting.**  
                    - **Do not repeat the schema.**  
                    - **Only return the JSON output.**  
                    - Provide only a JSON object that validates against the following JSON schema:  

                    ```json
                    \'{{
                      "$schema": "https://json-schema.org/draft/2020-12/schema",
                      "type": "object",
                      "properties": \'{{
                        "answers": \'{{
                          "type": "array",
                          "items": \'{{
                            "type": "string",
                            "enum": ["A", "B", "C", "D", "E"]
                          }}\'
                        }}\'
                      }}\',
                      "required": ["answers"]
                    }}\'
                    ```
                    If your response does not exactly match this format, it will be considered **INCORRECT**.  
                    {end_tokens}"""

        case "de":
            return """

                    {begin_tokens}
                    Stelle dir vor, du bist ein erfahrener Arzt und musst eine medizinische Frage beantworten.  
                    Die Antwortmöglichkeiten sind vorgegeben.  

                    ** Kontext: ** {context}

                    **Frage:** {question}  
                    **Antwortmöglichkeiten:** {options}  

                    ### Wichtige Anweisung:
                    -  Deine Antwort **muss** ein **valider JSON-Output** sein.  
                    - **Keine zusätzlichen Texte, Erklärungen oder Formatierungen.**  
                    - **Keine Wiederholung des Schemas.**  
                    - **Gib nur den JSON-Output zurück.**  
                    - Gib ausschließlich ein JSON-Objekt welches gegen das folgende JSON-Schema validiert:  


                    ```json
                    \'{{
                      "$schema": "https://json-schema.org/draft/2020-12/schema",
                      "type": "object",
                      "properties": \'{{
                        "answers": \'{{
                          "type": "array",
                          "items": \'{{
                            "type": "string",
                            "enum": ["A", "B", "C", "D", "E"]
                          }}\'
                        }}\'
                      }}\',
                      "required": ["answers"]
                    }}\'
                    ```
                    Falls deine Antwort nicht exakt diesem Format entspricht, gilt sie als **FALSCH**.
                    {end_tokens}"""


def get_prompt_text(language):
    match language:
        case "en":
            return """
                    {begin_tokens}
                    Imagine you are an experienced doctor and need to answer a medical question.  
                    The answer choices are provided.  
                
                    **Question:** {question}  
                    **Answer choices:** {options}  
                
                    ### Important Instructions:
                    - Your response **must** be a **valid JSON output**.  
                    - **No additional text, explanations, or formatting.**  
                    - **Do not repeat the schema.**  
                    - **Only return the JSON output.**  
                    - Provide only a JSON object that validates against the following JSON schema:  
                
                    ```json
                    \'{{
                      "$schema": "https://json-schema.org/draft/2020-12/schema",
                      "type": "object",
                      "properties": \'{{
                        "answers": \'{{
                          "type": "array",
                          "items": \'{{
                            "type": "string",
                            "enum": ["A", "B", "C", "D", "E"]
                          }}\'
                        }}\'
                      }}\',
                      "required": ["answers"]
                    }}\'
                    ```
                    If your response does not exactly match this format, it will be considered **INCORRECT**.  
                    {end_tokens}"""

        case "de":
            return """
        
                    {begin_tokens}
                    Stelle dir vor, du bist ein erfahrener Arzt und musst eine medizinische Frage beantworten.  
                    Die Antwortmöglichkeiten sind vorgegeben.  
        
                    **Frage:** {question}  
                    **Antwortmöglichkeiten:** {options}  
        
                    ### Wichtige Anweisung:
                    -  Deine Antwort **muss** ein **valider JSON-Output** sein.  
                    - **Keine zusätzlichen Texte, Erklärungen oder Formatierungen.**  
                    - **Keine Wiederholung des Schemas.**  
                    - **Gib nur den JSON-Output zurück.**  
                    - Gib ausschließlich ein JSON-Objekt welches gegen das folgende JSON-Schema validiert:  
        
        
                    ```json
                    \'{{
                      "$schema": "https://json-schema.org/draft/2020-12/schema",
                      "type": "object",
                      "properties": \'{{
                        "answers": \'{{
                          "type": "array",
                          "items": \'{{
                            "type": "string",
                            "enum": ["A", "B", "C", "D", "E"]
                          }}\'
                        }}\'
                      }}\',
                      "required": ["answers"]
                    }}\'
                    ```
                    Falls deine Antwort nicht exakt diesem Format entspricht, gilt sie als **FALSCH**.
                    {end_tokens}"""
        case "pt":
            return """
                {begin_tokens}
                Imagina que és um médico experiente e precisas de responder a questões médicas.
                As respostas possíveis sao providenciadas.

                **Questão:** {question}  
                **Respostas possíveis:** {options}  

                ### Instruções importantes:
                - A resposta **tem** de ser um **output json válido**..  
                - **Nada de texto adicional, explicações ou formato.**  
                - **Não repitas o schema.**  
                - **Retorna apenas o resultado JSON.**  
                - Retorna apenas um objecto JSON válido quando comparado com o seguinte esquema JSON:  

                ```json
                \'{{
                  "$schema": "https://json-schema.org/draft/2020-12/schema",
                  "type": "object",
                  "properties": \'{{
                    "answers": \'{{
                      "type": "array",
                      "items": \'{{
                        "type": "string",
                        "enum": ["A", "B", "C", "D", "E"]
                      }}\'
                    }}\'
                  }}\',
                  "required": ["answers"]
                }}\'
                ```
                Se a resposta não corresponda exactamente ao esquema indicado, vai ser considerado **ERRADA**.  
                {end_tokens}"""


def get_prompt_rag_template(language="de"):
    prompt_template_text = get_prompt_rag_text(language)

    prompt_template = PromptTemplate(
        template=prompt_template_text,
        input_variables=["context", "question", "options", "begin_tokens", "end_tokens"],
    )

    return prompt_template


def get_prompt_template(language="de"):
    prompt_template_text = get_prompt_text(language)

    prompt_template = PromptTemplate(
        template=prompt_template_text,
        input_variables=["question", "options", "begin_tokens", "end_tokens"],
    )

    return prompt_template
