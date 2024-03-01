from sklearn.metrics import precision_score, recall_score, f1_score
import torch


def evaluate_f1(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    f1 = f1_score(y_true, y_pred, average='micro')

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)


def test(model, tokenizer, x_test, y_test, id2label, label2id, device):
    correct = 0
    y_pred = []
    for text in x_test:
        inputs = tokenizer.encode(text, return_tensors="pt").to(device)
        logits = model(inputs).logits
        predictions = torch.max(logits, 1).indices
        prediction: str = id2label[predictions.tolist()[0]]
        actual: str = y_test[x_test.index(text)]
        y_pred.append(prediction)
        if prediction == actual:
            correct += 1
            print(f"Predicted Correctly: {prediction}, for: {text}")
        else:
            print(f"Predicted Wrongly: {
                  prediction} - Actual: {actual}, for: {text}")
    accuracy = correct / len(x_test)

    print(f'Accuracy: {accuracy}')
    return y_pred
