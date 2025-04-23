from evaluation import calculate_acc
from tabulate import tabulate
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import os
import json
matplotlib.use('TkAgg')

all_lang = ["de", "pt"]
all_prompt_lang = ["de", "en", "pt"]


def load_results(path="results"):
    res = os.listdir(path)
    if '.gitkeep' in res: res.remove('.gitkeep')
    if '.DS_Store' in res: res.remove('.DS_Store')
    return sorted(res)


results_path = load_results()
results_rag_path = load_results("rag")


def validate_one_lang(lang):
    if lang in all_lang:
        return lang
    else:
        return 'de'


def validate_one_prompt_lang(prompt_lang):
    if prompt_lang in all_prompt_lang:
        return prompt_lang
    else:
        return 'en'


def validate_multiple_lang(lang):
    if lang == "all":
        return all_lang
    if lang in all_lang:
        return [lang]
    elif all([l in all_lang for l in lang]):
        return lang
    else:
        return ['de']


def validate_multiple_prompt_lang(prompt_lang):
    if prompt_lang == "all":
        return all_prompt_lang
    elif prompt_lang in all_prompt_lang:
        return [prompt_lang]
    elif all([l in all_prompt_lang for l in prompt_lang]):
        return prompt_lang
    else:
        return ['de']


def print_model_results_by_prompt_language(model, prompt_lang="en"):
    prompt_lang = validate_one_prompt_lang(prompt_lang)

    table = []
    for p in results_path:

        with open('results/' + p, 'r') as in_file:
            data = json.load(in_file)
            if not (prompt_lang == data["language_prompt"].lower() and model == data["model"]):
                continue

            m, s, c, acc, non_acc = calculate_acc(data, False)
            table.append([s, acc, non_acc, c, data['duration']])

    print()
    print('Performance of ' + model + ' with the prompt in ' + prompt_lang)
    print(tabulate(table, headers=['Subject', 'Accuracy', 'Sum Correct Answers', 'Sum Questions', 'Duration']))


def print_model_results_by_language(model, lang="de"):
    lang = validate_one_lang(lang)

    table = []
    for p in results_path:

        with open('results/' + p, 'r') as in_file:
            data = json.load(in_file)
            if not (lang == data["language"].lower() and model == data["model"]):
                continue

            m, s, c, acc, non_acc = calculate_acc(data, False)
            table.append([s, acc, non_acc, c, data['duration']])

    print()
    print('Performance of ' + model + ' with questions in ' + lang)
    print(tabulate(table, headers=['Subject', 'Accuracy', 'Sum Correct Answers', 'Sum Questions', 'Duration']))


def print_single_model_results_for_lang(model, lang="de", prompt_lang="en"):
    lang_analyse = validate_one_lang(lang)
    prompt_lang_analyse = validate_one_prompt_lang(prompt_lang)

    if isinstance(model, list):
        model = model[0]

    results = {}  # Dictionary zur Speicherung der Ergebnisse pro Fach und Sprache

    for p in results_path:
        with open('results/' + p, 'r') as in_file:
            data = json.load(in_file)

            if not (model == data["model"] and
                    data["language"].lower() == lang_analyse and
                    data["language_prompt"].lower() in prompt_lang_analyse):
                continue

            m, subj, c, acc, non_acc = calculate_acc(data, False)

            if subj not in results:
                results[subj] = {}

            results[subj][data["language_prompt"].lower()] = [acc, non_acc, c, data['duration']]

    # Tabelle erstellen
    table = []
    for subject, languages in sorted(results.items()):
        row = [subject]
        for lang in prompt_lang_analyse:
            if lang in languages:
                row.extend(languages[lang])
            else:
                row.extend(['-', '-', '-', '-'])  # Falls für eine Sprache keine Daten existieren
        table.append(row)

    headers = ['Subject']

    topics = ['Accuracy (', 'Correct (', 'Total (', 'Duration (']
    for lang in prompt_lang_analyse:
        headers.extend([topic+lang.upper()+")" for topic in topics])

    # Calculate overall weighted avg acc and time
    results_avg = {l: [0, 0] for l in prompt_lang_analyse}
    nb_questions = 0

    for subject, languages in sorted(results.items()):
        nb_questions = nb_questions + results[subject][lang][2]

        for lang in prompt_lang_analyse:
            results_avg[lang][0] = results_avg[lang][0] + results[subject][lang][1]  # count nb corrected guesses
            results_avg[lang][1] = results_avg[lang][1] + results[subject][lang][3]  # count total time guesses

    print()
    print(f'Performance of {model} for {lang} exams with prompts in {(", ".join(prompt_lang_analyse)).upper()}')
    print(f'Overall results:\n' + "\n".join([f"{l.upper()} avg_acc={results_avg[l][0] / nb_questions:.3f} avg_time={results_avg[l][1] / nb_questions:.3f}s" for l in prompt_lang_analyse]))

    print(tabulate(table, headers=headers))


def print_model_comparison(models, lang='de', prompt_lang="de", print_duration=False):
    lang_analyse = validate_one_lang(lang)
    prompt_lang_analyse = validate_multiple_prompt_lang(prompt_lang)

    table, header = generate_comparison_table(models, lang=lang_analyse, prompt_lang=prompt_lang_analyse,
                                              print_duration=print_duration)

    print()
    print(f'Performance comparison of {" vs. ".join(models)} for {lang_analyse.upper()} exams using {", ".join([l.upper() for l in prompt_lang_analyse])} prompts')
    print(tabulate(table, headers=header))


def generate_comparison_table(models="all", lang='de', prompt_lang="en", print_duration=False):
    lang_analyse = validate_one_lang(lang)
    prompt_lang_analyse = validate_multiple_prompt_lang(prompt_lang)

    nb_questions = {l: 0 for l in prompt_lang_analyse}
    subjects_analysed = {l: [] for l in prompt_lang_analyse}

    results = {model: {} for model in models}  # Ergebnisse für alle Modelle speichern

    for p in results_path:
        with open('results/' + p, 'r') as in_file:
            data = json.load(in_file)

            if not (data["model"] in models and
                    data["language"].lower() == lang_analyse and
                    data["language_prompt"].lower() in prompt_lang_analyse):
                continue

            lang_key = data["language_prompt"].lower()
            model, s, count, acc, non_acc, correct_quest = calculate_acc(data, False)

            if s not in results[model]:
                results[model][s] = {}

            results[model][s][lang_key] = [acc, data['duration'], count, correct_quest]

            if s not in subjects_analysed[lang_key]:
                subjects_analysed[lang_key].append(s)
                nb_questions[lang_key] = nb_questions[lang_key] + count

    # Tabelle erstellen
    table = []
    subjects = sorted(set().union(*[results[model].keys() for model in models]))

    results_count = {model: {l: 0 for l in prompt_lang_analyse} for model in models}
    results_count_time = {model: {l: 0 for l in prompt_lang_analyse} for model in models}

    # Count correct answers over all
    for model in models:
        for subject in subjects:
            for l in prompt_lang_analyse:
                if subject in results[model].keys() and l in results[model][subject].keys():
                    results_count[model][l] = results_count[model][l] + results[model][subject][l][3]
                    results_count_time[model][l] = results_count_time[model][l] + results[model][subject][l][1]

    row = [" - Overall Average - "]
    remove_lang = set()
    for model in models:
        for l in prompt_lang_analyse:
            # Delete unused language
            if nb_questions[l] == 0:
                results_count_time[model].pop(l)
                results_count[model].pop(l)
                remove_lang.add(l)
                continue

            row.append(results_count[model][l] / nb_questions[l])
            row.append(results_count[model][l])
            row.append(nb_questions[l])

            if print_duration:
                row.append(results_count_time[model][l] / nb_questions[l])

    [prompt_lang_analyse.remove(l) for l in remove_lang]
    [nb_questions.pop(l) for l in remove_lang]

    table.append(row)

    for subject in subjects:
        row = [subject]
        for model in models:
            if subject in results[model].keys():
                for l in prompt_lang_analyse:
                    if not l in results[model][subject].keys():
                        continue
                    results_count[model][l] = results_count[model][l] + results[model][subject][l][-1]

                    row.append(results[model][subject].get(l, ['-', '-', '-', '-'])[0])  # Acc
                    row.append(results[model][subject].get(l, ['-', '-', '-', '-'])[3])  # Correct Answers
                    row.append(results[model][subject].get(l, ['-', '-', '-', '-'])[2])  # Nb Answers
                    if print_duration:
                        row.append(results[model][subject].get(l, ['-', '-', '-', '-'])[1])  # Duration

            else:
                for _ in prompt_lang_analyse:
                    row.append('-')  # Acc Platzhalter
                    row.append('-')  # Correct Answers Platzhalter
                    row.append('-')  # Nb Answers Platzhalter
                    if print_duration:
                        row.append('-')  # Duration Platzhalter

        table.append(row)

    # Überschriften generieren
    header = ['Subject']
    for model in models:
        for l in prompt_lang_analyse:
            header.append(f'Acc ({model}-{l.upper()})')
            header.append(f'Correct Answers ({model}-{l.upper()})')
            header.append(f'Nb Answers ({model}-{l.upper()})')
            if print_duration:
                header.append(f'Duration ({model}-{l.upper()})')

    return table, header


def get_tested_models(to_print=False):
    tested_models = set()

    for r in results_path:
        tested_models.add(r.split('_')[0])

    if to_print:
        print(tested_models)

    return tested_models


def get_subject_group(lang):
    if lang == "de":
        return {"Surgical Specialties": ["Chirurgie", "Orthopaedie", "Urologie"],
                "Obstetrics & Gynecology": ["Frauenheilkunde", "Frauenheilkunde_Musterklausur"],
                "Diagnostic & Interventional Medicine": ["Radiologie", "Strahlentherapie"],
                "Internal Medicine": ["Innere-Diabeto-Endokrino-Nephro", "Innere-Gastro-Onko-Haemo",
                                      "Innere-Kardio-Pulmo", "Infektiologie-Immunologie"],
                "Foundational Medicine & Forensics": ["AllgemeinePathologie", "Rechtsmedizin",
                                                      "KlinischePharmakologie", "AllgemeinePharmakologie",
                                                      "KlinischeChemie"],
                "Psychiatry & Neurology": ["Neurologie", "Psychiatrie", "Sozialmedizin"],
                "Others": ["Augenheilkunde", "Dermatologie", "Humangenetik", "HNO", "Anaesthesiologie",
                           "Notfallmedizin", "Kinderheilkunde"]
                }
    elif lang == "pt":
        return {"Bio-electricity": ["Biom_Bioelectricidade_1"],
                "Gynecology": ["Medicine_Ginecologia_1"],
                "Anatomy & Histology": ["Biom_Anatomia_Histo_1", "Biom_Anatomia_Histo_2", "Biom_Anatomia_Histo_3"],
                "General Mechanisms of Disease": ["Biom_MGD_1", "Biom_MGD_3", "Biom_MGD_4", "Biom_MGD_5"],
                "Broad Exam": ["Medicine_Integrado_1"],
                "Pediatrics": ["Medicine_Pediatria"],
                "Psychiatry": ["Medicine_Psiquiatria_1"]
                }
    else:
        return [], [[]]


def plot_by_family(lang="de", prompt_lang="en", other=False, save=False):
    lang_analyse = validate_one_lang(lang)
    prompt_lang_analyse = validate_one_prompt_lang(prompt_lang)

    # {family: {model : {name : X , nb_param: Y} } }

    models_family = {
        "gemma3": {
            'gemma3:1b': {"name": "gemma3:1b", "nb_param": 1},
            'gemma3:4b': {"name": "gemma3:4b", "nb_param": 4},
            'gemma3:12b': {"name": "gemma3:12b", "nb_param": 12},
            'gemma3:27b': {"name": "gemma3:27b", "nb_param": 27}
        },
        "llama": {
            'llama3:70b': {"name": "llama3:70b", "nb_param": 70},
            'llama3.1:8b': {"name": "llama3.1:8b", "nb_param": 8},
            'llama3.1:70b': {"name": "llama3.1:70b", "nb_param": 70},
            'llama3.2:latest': {"name": "llama3.2:latest", "nb_param": 3},
            'llama3.3:latest': {"name": "llama3.3:latest", "nb_param": 70},
            # llama4
        },
        "mistral": {
            'mistral:latest': {"name": "mistral:latest", "nb_param": 7},
            'mistral-nemo:latest': {"name": "mistral-nemo:latest", "nb_param": 12},
            'mistral-large:latest': {"name": "mistral-large:latest", "nb_param": 123},
        },
        "qwen": {
            'qwen:0.5b': {"name": "qwen:0.5b", "nb_param": 0.5},
            'qwen:1.8b': {"name": "qwen:1.8b", "nb_param": 1.8},
            'qwen:4b': {"name": "qwen:4b", "nb_param": 4},
            'qwen:7b': {"name": "qwen:7b", "nb_param": 7},
            'qwen:14b': {"name": "qwen:14b", "nb_param": 14},
            'qwen:32b': {"name": "qwen:32b", "nb_param": 32},
            'qwen:72b': {"name": "qwen:72b", "nb_param": 72},
            'qwen:110b': {"name": "qwen:110b", "nb_param": 110},
        },
        "qwen2.5": {
            'qwen2.5:0.5b': {"name": "qwen2.5:0.5b", "nb_param": 0.5},
            'qwen2.5:1.5b': {"name": "qwen2.5:1.5b", "nb_param": 1.5},
            'qwen2.5:3b': {"name": "qwen2.5:3b", "nb_param": 3},
            'qwen2.5:7b': {"name": "qwen2.5:7b", "nb_param": 7},
            'qwen2.5:14b': {"name": "qwen2.5:14b", "nb_param": 14},
            'qwen2.5:32b': {"name": "qwen2.5:32b", "nb_param": 32},
            'qwen2.5:72b': {"name": "qwen2.5:72b", "nb_param": 72},
        },
        "other": {
            # 'gpt-4o': {"name": "gpt-4o", "nb_param": 0},  # Not disclosed 500B to 2/3T
            'phi4:latest': {"name": "phi4:latest", "nb_param": 5.6},
            'deepseek-r1:70b': {"name": "deepseek-r1:70b", "nb_param": 70},
            'qwq:latest': {"name": "qwq", "nb_param": 32},
        }
    }

    if not other:
        models_family.pop("other")

    table, header = generate_comparison_table(get_tested_models(), lang=lang_analyse, prompt_lang=prompt_lang_analyse,
                                              print_duration=False)

    # Get only overall average and corresponding model from table and header
    table = [elem * 100 for elem in table[0][1::3]]
    header = header[1::3]
    header = [entry.split('(')[1].split(f"-{prompt_lang_analyse.upper()}")[0] for entry in header]

    dict_results = {header[i]: table[i] for i in range(0, len(table))}

    # Plotting
    plt.figure(figsize=(5, 4))

    for family, models in models_family.items():
        x_params = []
        y_accuracies = []

        for model_name, info in models.items():
            if model_name in dict_results:
                x_params.append(info['nb_param'])
                y_accuracies.append(dict_results[model_name])

        if x_params:  # only plot if there's something to show
            # Sort by number of parameters
            sorted_pairs = sorted(zip(x_params, y_accuracies))
            x_sorted, y_sorted = zip(*sorted_pairs)

            plt.plot(x_sorted, y_sorted, marker='o', label=family)

    # Maximum - gpt-4o
    plt.axhline(y=dict_results["gpt-4o"], color='black', linestyle='--', linewidth=1)
    plt.text(x=5, y=dict_results["gpt-4o"] + 1,
             s="GPT-4o",
             fontsize=10,
             color='gray'
             )

    # Plot config
    plt.xlabel(f"Number of Parameters (B)")
    plt.ylabel("Accuracy")
    plt.title(f"{lang.upper()} - Model Accuracy vs. Size by Family (B)")
    plt.ylim(0, 100)
    plt.xlim(0, 130)
    plt.legend(title="Model Family")
    plt.grid(True)
    plt.tight_layout()

    # Display
    plt.tight_layout()
    if save:
        plt.savefig(f"images/family_analysis_{lang_analyse}_prompt_{prompt_lang_analyse.upper()}_{'T' if other else 'F'}.png", dpi=300,
                    bbox_inches='tight')
    else:
        plt.show()


def print_model_results_by_language_and_prompt(model, lang="de", prompt_lang="en"):
    lang = validate_one_lang(lang)
    prompt_lang = validate_one_prompt_lang(prompt_lang)

    table = []
    for p in results_path:

        with open('results/' + p, 'r') as in_file:
            data = json.load(in_file)
            if not (lang == data["language"].lower() and model == data["model"] and prompt_lang == data["language_prompt"].lower()):
                continue

            m, s, c, acc, non_acc = calculate_acc(data, False)
            table.append([s, acc, non_acc, c, data['duration']])
    print(tabulate(table, headers=['Subject', 'Accuracy', 'Sum Correct Answers', 'Sum Questions', 'Duration']))

    print()
    print('Performance of ' + model + ' with questions in ' + lang + " and prompt in " + prompt_lang)


def plot_models_results(models, lang='de', prompt_lang="en", save=False, title=""):
    lang_analyse = validate_one_lang(lang)
    prompt_lang_analyse = validate_one_prompt_lang(prompt_lang)

    table, header = generate_comparison_table(models, lang=lang_analyse, prompt_lang=prompt_lang_analyse,
                                              print_duration=False)

    group_subjects = get_subject_group(lang_analyse)
    # table_clean = [[row[0]] + row[1::3] for row in table]
    # header_clean = [header[0]] + [model.split("(")[1].split(f"-{prompt_lang_analyse.upper()}")[0] for model in header[1::3]]

    # # Group Subjects
    # Convert table to dict for easier lookup and manipulation
    table_dict = {row[0]: row[1:] for row in table if row[0] != ' - Overall Average - '}

    # New grouped rows
    overall_avg = table[0][1::3]
    final_table = [['Overall Average'] + overall_avg]

    nb_models = len(models)

    # Grouping
    for group_name, subjects in group_subjects.items():
        total_correct = [0]*nb_models
        total_questions = [0]*nb_models

        # Sum values
        for subj in subjects:
            if subj in table_dict:
                correct, questions = table_dict[subj][1::3], table_dict[subj][2::3]

                total_correct = [x + y for x, y in zip(total_correct, correct)]
                total_questions = [x + y for x, y in zip(total_questions, questions)]
                del table_dict[subj]  # Remove from original

        avgs = [x / y for x, y in zip(total_correct, total_questions)]
        final_table.append([group_name] + avgs)

    # # # # # #  Generate Radar Graph
    # Get data from table
    header_clean = [header[0]] + [model.split("(")[1].split(f"-{prompt_lang_analyse.upper()}")[0] for model in
                                  header[1::3]]
    table_clean = [[row[0]] + [elem * 100 for elem in row[1:]] for row in final_table]

    # Convert to DataFrame
    df = pd.DataFrame(table_clean, columns=header_clean)
    df.set_index('Subject', inplace=True)

    # Reverse order for subjects (Show overall on top)
    df = df[::-1]

    # Plot
    df.plot(kind='barh', figsize=(10, 7), width=0.8, colormap='coolwarm_r')
    plt.ylabel('Subject', fontsize=12)
    plt.xlabel('Accuracy', fontsize=12)
    if title:
        plt.title(title, fontsize=14)
    else:
        plt.title(f'{lang_analyse.upper()} Exams - Accuracy Comparison Across Models', fontsize=14)

    plt.yticks(rotation=0, fontsize=10)
    plt.xlim(0, 100)
    plt.legend(title="Models", fontsize=10, reverse=True)
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    if save:
        plt.savefig(f"images/models_comparison_{lang_analyse}_prompt_{prompt_lang_analyse}.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()


def print_model_radar_chart(model, lang="de", prompt_lang="en", save=False):
    lang_analyse = validate_one_lang(lang)
    prompt_lang_analyse = validate_multiple_prompt_lang(prompt_lang)

    table, header = generate_comparison_table([model], lang=lang_analyse, prompt_lang=prompt_lang_analyse,
                                              print_duration=False)

    group_subjects = get_subject_group(lang_analyse)
    labels = []
    values = []
    shift = -3

    # Loop in case we want more than one prompt language
    for _ in prompt_lang_analyse:
        shift = shift + 3
        # # Group Subjects
        # Convert table to dict for easier lookup and manipulation
        table_dict = {row[0]: row[1:] for row in table if row[0] != ' - Overall Average - '}

        # New grouped rows
        overall_avg = table[0][shift + 1]
        final_table = [['Overall Average', overall_avg]]

        # Grouping
        for group_name, subjects in group_subjects.items():
            total_correct = 0
            total_questions = 0

            # Sum values
            for subj in subjects:
                if subj in table_dict:
                    correct, total = table_dict[subj][shift + 1], table_dict[subj][shift + 2]

                    total_correct += correct
                    total_questions += total
                    del table_dict[subj]  # Remove from original

            # Avoid division by zero
            avg = total_correct / total_questions if total_questions else 0
            final_table.append([group_name, avg])

        # # # # # #  Generate Radar Graph
        # Get data from table
        labels.append([entry[0] for entry in final_table])
        values.append([entry[1] * 100 for entry in final_table])

        # Radar chart requires the values to be circular
        values[-1] += values[-1][:1]  # Repeat the first value at the end

    labels = labels[0]
    num_vars = len(labels)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw the outline and fill
    ax.plot(angles, values[0], color='tab:blue', linewidth=2, zorder=1)
    if values.__len__() > 1:
        ax.plot(angles, values[1], color='tab:red', linewidth=2, zorder=1)

    ax.fill(angles, values[0], color='tab:blue', alpha=0.25, zorder=1)
    if values.__len__() > 1:
        ax.fill(angles, values[1], color='tab:red', alpha=0.25, zorder=1)

    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    for label in ax.get_xticklabels():
        label.set_fontsize(11)
        label.set_fontweight('bold')
        label.set_zorder(10)

    # Set y-labels style
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=10)
    ax.yaxis.grid(True, color="gray", linestyle='dashed', linewidth=0.5)

    # Add grid and circular frame
    ax.xaxis.grid(True, linestyle='dashed', linewidth=0.5)
    ax.spines['polar'].set_visible(True)
    ax.spines['polar'].set_color('gray')
    ax.spines['polar'].set_linewidth(1)

    # Optional title
    plt.title(f'{model} - Subject-wise Performance Overview - {lang_analyse.upper()}', size=16, fontweight='bold', pad=20)

    # Display
    plt.tight_layout()
    if save:
        plt.savefig(f"images/radar_chart_{model}_{lang_analyse}_prompt_{'_'.join(prompt_lang_analyse)}.png", dpi=300,
                    bbox_inches='tight')
    else:
        plt.show()


def generate_rag_comparison_table(model="llama3.3:latest", lang='de', prompt_lang="en", print_duration=False):
    lang_analyse = validate_one_lang(lang)
    prompt_lang_analyse = validate_one_prompt_lang(prompt_lang)

    nb_questions = 0
    results_rag = {}
    results = {}

    for p in results_rag_path:
        # Load RAG data
        with open('rag/' + p, 'r') as in_file:
            data = json.load(in_file)

        if not (data["model"] == model and
                data["language"].lower() == lang_analyse and
                data["language_prompt"].lower() in prompt_lang_analyse):
            continue

        _, s, count, acc, non_acc, correct_quest = calculate_acc(data, False)

        results_rag[s] = [acc, data['duration'], count, correct_quest]

        # Load equivalent non RAG data
        with open('results/' + p, 'r') as in_file:
            data = json.load(in_file)

        _, s, count, acc, non_acc, correct_quest = calculate_acc(data, False)

        results[s] = [acc, data['duration'], count, correct_quest]

        nb_questions = nb_questions + count

    if nb_questions == 0:
        return [], []

    subjects = list(results_rag.keys())

    # Calculate overall results
    results_rag_count = 0
    results_count = 0
    for subj in subjects:
        results_rag_count += results_rag[subj][3]
        results_count += results[subj][3]

    results_rag_count_time = 0
    results_count_time = 0
    for subj in subjects:
        results_rag_count_time += results_rag[subj][1]
        results_count_time += results[subj][1]

    table = [[" - Overall Average - ", results_rag_count / nb_questions, results_rag_count, nb_questions,
              results_count / nb_questions, results_count, nb_questions]]

    if print_duration:
        table[0] += [results_rag_count_time / nb_questions, results_count_time / nb_questions]

    for subject in subjects:
        results_rag_row = results_rag[subject]
        results_row = results[subject]
        row = [subject]
        for row_aux in [results_rag_row, results_row]:
            row.append(row_aux[0])  # Acc
            row.append(row_aux[3])  # Correct Answers
            row.append(row_aux[2])  # Nb Answers
            if print_duration:
                row.append(row_aux[1])  # Duration

        table.append(row)  # TODO: check this part

    # Überschriften generieren
    header = ['Subject', 'RAG - Acc', 'RAG - Correct Answers', 'RAG - Nb Answers']
    if print_duration:
        header.append(f'RAG - Duration')

    header += ['Acc', 'Correct Answers', 'Nb Answers']
    if print_duration:
        header.append(f'Duration')

    return table, header

def plot_model_rag_radar_chart(model, lang="de", prompt_lang="en", save=False, title=""):
    lang_analyse = validate_one_lang(lang)
    prompt_lang_analyse = validate_one_prompt_lang(prompt_lang)

    table, header = generate_rag_comparison_table(model, lang=lang_analyse, prompt_lang=prompt_lang_analyse,
                                                  print_duration=False)

    group_subjects = get_subject_group(lang_analyse)
    labels = []
    values = []

    # Loop get RAG and non RAG
    for shift in [0, 3]:
        # # Group Subjects
        # Convert table to dict for easier lookup and manipulation
        table_dict = {row[0]: row[1:] for row in table if row[0] != ' - Overall Average - '}

        # New grouped rows
        overall_avg = table[0][shift + 1]
        final_table = [['Overall Average', overall_avg]]

        # Grouping
        for group_name, subjects in group_subjects.items():
            total_correct = 0
            total_questions = 0

            # Sum values
            for subj in subjects:
                if subj in table_dict:
                    correct, total = table_dict[subj][shift + 1], table_dict[subj][shift + 2]

                    total_correct += correct
                    total_questions += total
                    del table_dict[subj]  # Remove from original

            # Avoid division by zero
            avg = total_correct / total_questions if total_questions else 0
            final_table.append([group_name, avg])

        # # # # # #  Generate Radar Graph
        # Get data from table
        labels.append([entry[0] for entry in final_table])
        values.append([entry[1] * 100 for entry in final_table])

        # Radar chart requires the values to be circular
        values[-1] += values[-1][:1]  # Repeat the first value at the end

    labels = labels[0]
    num_vars = len(labels)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # Create plot
    fig, ax = plt.subplots(figsize=(7.1, 7.1), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw the outline and fill
    ax.plot(angles, values[0], color='tab:blue', linewidth=2, zorder=1)
    if values.__len__() > 1:
        ax.plot(angles, values[1], color='tab:red', linewidth=2, zorder=1)

    ax.fill(angles, values[0], color='tab:blue', alpha=0.25, zorder=1)
    if values.__len__() > 1:
        ax.fill(angles, values[1], color='tab:red', alpha=0.25, zorder=1)

    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    for label in ax.get_xticklabels():
        label.set_fontsize(11)
        label.set_fontweight('bold')
        label.set_zorder(10)

    # Set y-labels style
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=10)
    ax.yaxis.grid(True, color="gray", linestyle='dashed', linewidth=0.5)

    # Add grid and circular frame
    ax.xaxis.grid(True, linestyle='dashed', linewidth=0.5)
    ax.spines['polar'].set_visible(True)
    ax.spines['polar'].set_color('gray')
    ax.spines['polar'].set_linewidth(1)

    # Optional title
    if title:
        plt.title(title, size=16, fontweight='bold', pad=20)
    else:
        plt.title(f'{model} - RAG Performance Overview ({prompt_lang_analyse.upper()})', size=16, fontweight='bold', pad=20)

    # Display
    plt.tight_layout()
    if save:
        plt.savefig(f"images/RAG_{model}_{lang_analyse}_prompt_{prompt_lang_analyse}.png", dpi=300,
                    bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':
    # models = get_tested_models()

    # Plot RAG
    plot_model_rag_radar_chart(model="llama3.3:latest", lang="de", prompt_lang="en", save=True, title="RAG Performance Overview")
    plot_model_rag_radar_chart(model="llama3.3:latest", lang="de", prompt_lang="de", save=True)

    # # Plot by family
    # plot_by_family(lang="de", prompt_lang="en", other=False, save=True)
    # plot_by_family(lang="de", prompt_lang="en", other=True, save=True)
    # plot_by_family(lang="de", prompt_lang="de", other=False, save=True)
    # plot_by_family(lang="pt", prompt_lang="en", other=False, save=True)
    # plot_by_family(lang="pt", prompt_lang="en", other=True, save=True)
    # plot_by_family(lang="pt", prompt_lang="pt", other=False, save=True)

    # # Radar Plot
    # print_model_radar_chart("llama3.3:latest", lang="pt", prompt_lang=["en", "pt"], save=True)
    # print_model_radar_chart("llama3.3:latest", lang="de", prompt_lang=["en", "de"], save=True)
    # print_model_radar_chart("gpt-4o", lang="pt", prompt_lang=["en", "pt"], save=True)
    # print_model_radar_chart("gpt-4o", lang="de", prompt_lang=["en", "de"], save=True)
    # print_model_radar_chart("qwen2.5:72b", lang="pt", prompt_lang=["en", "pt"], save=True)
    # print_model_radar_chart("qwen2.5:72b", lang="de", prompt_lang=["en", "de"], save=True)

    # # Model Comparison
    # Models selected by hand, based on popularity or performance relative to respective family
    # models = ["gpt-4o", 'llama3.3:latest', "qwen2.5:72b", "mistral-large:latest", "gemma3:27b", 'llama3.1:70b',
    #           "qwq:latest", 'llama3:70b', "deepseek-r1:70b"]
    # plot_models_results(models, lang='en', prompt_lang="en", save=True, title="Model Comparison")
    # plot_models_results(models, lang='en', prompt_lang="de", save=True)
    # plot_models_results(models, lang='pt', prompt_lang="en", save=True)
    # plot_models_results(models, lang='pt', prompt_lang="pt", save=True)
