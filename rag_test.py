from tqdm import tqdm
from utils.pipeline import *
from dotenv import load_dotenv
from datetime import datetime
from utils.evaluation import calculate_acc
from llama_index.retrievers.bm25 import BM25Retriever
from utils.rag_documents import get_nodes_by_subject, get_document_by_subject

import os
import json

load_dotenv()

INPUT_PATH = 'curated/'
OUT_PATH = "rag/"


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

        if len(data['subject']) > 0:
            temp[data['subject']] = data
    return temp


def get_set(in_data):
    i = in_data['idx']
    question = in_data['question']
    options = in_data['options']
    answer = in_data['answer']
    return i, question, options, answer


def take_exam(language, model, subject, questions):
    exams_results = dict()
    exams_results['model'] = model
    exams_results['subject'] = subject
    exams_results['language'] = subject.split('_', 1)[0]
    exams_results['language_prompt'] = language

    answers = dict()
    nodes = get_nodes_by_subject(subject)
    if nodes:
        receiver = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=3, language='de')
    else:
        receiver = None

    start = datetime.now()
    for entry in tqdm(questions, desc=subject):
        i, q, o, a = get_set(entry)

        if a == '###':
            continue

        if receiver:
            nodes = receiver.retrieve(q)
        else:
            nodes = []
        res = run_question_with_rag(language, model, nodes, q, o)
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


def run_rag_test():
    model = 'llama3.3:latest'

    exams = get_exams()
    model_dryrun(model)
    subjects = sorted(list(exams.keys()))

    for subject in subjects:
        for prompt_language in ['de', 'en']:

            file_path = OUT_PATH + model + "_" + prompt_language + "_" + str(subject).replace(' ', '') + '.json'

            if os.path.exists(file_path):
                print("Skipping " + file_path + " - Already exists")
                continue

            if not get_document_by_subject(subject):
                print("Skipping " + file_path + " - No book associated")
                continue


            print()
            results = take_exam(prompt_language, model, subject, exams[subject]['questions'])
            with open(file_path, 'w') as out:
                out.write(json.dumps(results, ensure_ascii=False))

            print(prompt_language + ": ")
            calculate_acc(results, True)


if __name__ == '__main__':
    run_rag_test()
