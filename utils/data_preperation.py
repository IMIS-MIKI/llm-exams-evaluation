import os
import json

IN_PATH = 'raw/'
OUT_PATH = 'exams/'


def format_text(text):
    text = text.replace('<strong>', '')
    text = text.replace('</strong>', '')
    text = text.replace('</ br>', '')
    text = text.replace('<br />', '')
    text = text.replace('<p>', ' ')
    text = text.replace('</p>', '')
    text = text.replace('<ol>', '')
    text = text.replace('</ol>', '')
    text = text.replace('<li>', '-')
    text = text.replace('</li>', '. ')

    return text


def get_file_paths():
    in_files = os.listdir(IN_PATH)
    if '.gitkeep' in in_files: in_files.remove('.gitkeep')
    if '.DS_Store' in in_files: in_files.remove('.DS_Store')
    return in_files


if __name__ == '__main__':

    paths = get_file_paths()
    answer_enum = ['A', 'B', 'C', 'D', 'E']

    # run over all raw files
    for p in paths:
        temp = dict()
        temp['subject'] = p.split('.')[0]
        temp['questions'] = list()
        temp['multiple_answers_allowed'] = False

        # Loading the raw files
        with open(IN_PATH + p, 'r') as in_file:
            data = json.load(in_file)

        entries = dict()

        # getting the answers
        # key is the question uuid
        for a in data['data']['answers']:
            question = data['data']['answers'][a]['fk_Question_question']
            answer_text = format_text(data['data']['answers'][a]['text'])

            if question in entries.keys():
                t = entries[question]
                t['options'].append(answer_text)
            else:
                entries[question] = {'options': [answer_text], 'answer': []}

            if data['data']['answers'][a]['correct'] == 1:
                c_temp = entries[question]['answer']
                c_temp.append(answer_text)
                entries[question]['answer'] = c_temp

        # formatting the answer from a list to dict with letters
        for question in entries:
            options = dict(zip(answer_enum, entries[question]['options']))
            entries[question]['options'] = options

            answer_literal = list()

            for o in entries[question]['options']:
                if entries[question]['options'][o] in entries[question]['answer']:
                    answer_literal.append(o)

            entries[question]['answer'] = answer_literal

            if len(answer_literal) > 1:
                temp['multiple_answers_allowed'] = True

        # getting the question text corresponding to the uuid
        for q in data['data']['questions']:
            entries[q]['question'] = format_text(data['data']['questions'][q]['text'])

        # combining all together
        temp['questions'] = list(entries.values())

        for idx, x in enumerate(temp['questions']):
            x['idx'] = idx + 1

        with open(OUT_PATH + temp['subject'] + '.json', 'w') as out:
            out.write(json.dumps(temp, ensure_ascii=False, sort_keys=True))
