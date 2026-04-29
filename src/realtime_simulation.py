import time

def simulate_realtime(model, X):
    print("\n--- Real-Time Stress Prediction ---")

    for i in range(len(X)):
        sample = X.iloc[i].values.reshape(1, -1)
        pred = model.predict(sample)

        if pred[0] == 0:
            level = "LOW"
        elif pred[0] == 1:
            level = "MEDIUM"
        else:
            level = "HIGH"

        print(f"Step {i}: Stress Level = {level}")
        time.sleep(1)