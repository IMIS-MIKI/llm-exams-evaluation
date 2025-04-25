from tqdm import tqdm
from utils.pipeline import *
from dotenv import load_dotenv
from datetime import datetime
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Settings

import os
import json

load_dotenv()

INPUT_PATH = 'curated/'
OUT_PATH = "results/"


def get_file_paths():
    exam_files = os.listdir(INPUT_PATH)
    if '.gitkeep' in exam_files: exam_files.remove('.gitkeep')
    if '.DS_Store' in exam_files: exam_files.remove('.DS_Store')
    return exam_files


def get_exams():
    paths = get_file_paths()
    temp = dict()

    for p in paths:
        with open(INPUT_PATH + p, 'r') as in_file:
            data = json.load(in_file)

        # Store with the name of the file without .json
        temp[p[:-5]] = data
    return temp


def get_set(in_data):
    i = in_data['idx']
    question = in_data['question']
    options = in_data['options']
    answer = in_data['answer']
    return i, question, options, answer


def take_exam(language, model, subject, questions, is_openai=False):
    exams_results = dict()
    exams_results['model'] = model
    exams_results['subject'] = subject.split('_', 1)[1]
    exams_results['language'] = subject.split('_', 1)[0]
    exams_results['language_prompt'] = language

    answers = dict()

    start = datetime.now()
    for entry in tqdm(questions, desc=subject):
        i, q, o, a = get_set(entry)

        if a == '###':
            continue

        res = run_question(language, model, q, o, is_openai)
        if len(res['outcomes'][0]) > 0:
            try:
                r = res['outcomes'][0]['answers']
            except:
                r = ["Z"]
        else:
            r = ["Z"]
        answers[i] = (a, r)

    end = datetime.now()

    exams_results['answers'] = answers
    exams_results['duration'] = (end - start).total_seconds()
    return exams_results


if __name__ == '__main__':
    #### OLLAMA #####
    models = ['mistral-large:latest', 'qwen:72b', 'qwen:110b', 'qwen:0.5b', 'qwen:1.8b', 'qwen:4b', 'qwen:7b', 'qwen:14b', 'qwen:32b' ]
    is_openai = False

    #### OPENAI #####
    # "gpt-4", "gpt-3.5-turbo"
    # models = ['gpt-4o']
    # is_openai = True

    exams = get_exams()
    subjects = sorted(list(exams.keys()))

    for model in models:
        print("Analysing Model: {}".format(model))
        model_loaded = False

        for language in ['en', "pt", "de"]:
            print("Language: {}".format(language))

            print()
            for subject in subjects:

                file_path = OUT_PATH + model + "_" + language + "_" + str(subject).replace(' ', '') + '.json'

                # Skip if result has already been calculated or language does not match (e.g. run de prompt for pt exam)
                if os.path.exists(file_path) or (language != "en" and not subject.startswith(language.upper())):
                    print("Skipping " + file_path)
                    continue

                # Dryrun to let ollama load the model on the GPU
                if not is_openai and not model_loaded:
                    print("Loading model: {}".format(model))
                    model_dryrun(model)
                    model_loaded = True

                try:
                    results = take_exam(language, model, subject, exams[subject]['questions'],
                                        is_openai=is_openai)
                    with open(file_path,
                              'w') as out:
                        out.write(json.dumps(results, ensure_ascii=False))

                except Exception as e:
                    print("Failed to take exam for " + subject + ": " + str(e))
                    continue


