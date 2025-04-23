from sklearn.metrics import accuracy_score


def calculate_acc(results, printing=True):
    y_pred = []
    y_true = []

    model = results['model']
    subject = results['subject']

    try:
        for entry in results['answers']:
            y_true.append(1)
            answer = results['answers'][entry][1]
            if not (isinstance(answer, list) and all(isinstance(item, str) for item in answer)):
                y_pred.append(0)
                continue

            solution = sorted(results['answers'][entry][0])
            answer = sorted(results['answers'][entry][1])
            # Check that answer is a list of strings
            y_pred.append(1 if solution == answer else 0)

    except:
        print(model, subject)


    non_normalize_acc = accuracy_score(y_true, y_pred, normalize=False)
    acc = accuracy_score(y_true, y_pred)
    question_count = len(y_pred)
    correct_questions = sum(y_pred)

    if printing:
        print(model + ' performed on ' + subject)
        print('Accuracy \t' + str(acc))
        print('Non normalized \t' + str(non_normalize_acc) + ' out of ' + str(question_count))
        print()

    return model, subject, question_count, acc, non_normalize_acc, correct_questions
