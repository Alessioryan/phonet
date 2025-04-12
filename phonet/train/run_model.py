import numpy as np
import tensorflow as tf
import pickle
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from phonet.train.main_train_RNN_MT import generate_data_test


def get_posteriorgram_custom(file_audio_feat, model_path, mu, std, feature_names):
    with open(file_audio_feat, 'rb') as f:
        data = pickle.load(f)
    features = data
    features = (features - mu) / std

    model = load_model(model_path)

    window_size = 40
    half_win = window_size // 2
    n_frames = features.shape[0]

    # Pad the sequence with zeros (or reflection, etc.) so we can center windows at edges
    padded = np.pad(features, ((half_win, half_win), (0, 0)), mode='edge')  # shape: (n+window_size, 34)

    # Create sliding windows centered on each original frame
    windows = np.stack([padded[i:i+window_size] for i in range(n_frames)])  # shape: (n, 40, 34)

    input_tensor = tf.convert_to_tensor(windows, dtype=tf.float32)  # shape: (n, 40, 34)

    # Predict — output could be list (multi-output model) or array (single-output)
    output = model.predict(input_tensor)  # shape: (n, num_feats)

    # If it's a list (multi-output), stack across last axis
    if isinstance(output, list):
        output = np.stack(output, axis=-1)  # shape: (n, num_outputs)

    print(output.shape)
    posteriorgram = output[:, 20, :, :]   # pick center timestep: (116, 2, 2)
    print(posteriorgram.shape)
    posteriorgram = np.diagonal(posteriorgram, axis1=-2, axis2=-1)  # shape: (n_frames, n_features, n_classes)
    print(posteriorgram.shape)

    return posteriorgram


# # Experiments with the models
# def experiments(model_path, file_feat_test, mu, std):
#     batch_size_val=1
#     Nfiles_test=len(os.listdir(file_feat_test))
#     validation_steps=int(Nfiles_test/batch_size_val)

#     # Load in the model
#     modelPH = load_model(model_path)
#     print(modelPH.input_shape)
#     ypred=modelPH.predict(generate_data_test(file_feat_test, batch_size_val, mu, std), steps=validation_steps)
#     print(ypred)


from scipy.io import wavfile

def plot_features_over_audio(wav_path, posteriorgram, feature_names=None, title="Features over audio"):
    import matplotlib.pyplot as plt
    from scipy.io import wavfile

    # Load audio
    sr, wav = wavfile.read(wav_path)
    wav = wav.astype(float)
    time_audio = np.linspace(0, len(wav) / sr, num=len(wav))

    # Generate time axis for posteriorgram
    num_frames = posteriorgram.shape[0]
    time_post = np.linspace(0, len(wav) / sr, num=num_frames)

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(time_audio, wav / np.max(np.abs(wav)), color='lightgray', label='Waveform')

    for i in range(posteriorgram.shape[1]):
        label = feature_names[i] if feature_names and i < len(feature_names) else f'Feature {i+1}'
        plt.plot(time_post, posteriorgram[:, i], label=label)

    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig("features_over_audio.png")
    # plt.show()




if __name__ == "__main__":
    print("Current working directory is:", os.getcwd())
    # Example usage
    file_audio_feat = '../features/test/071005a_2.pickle'
    file_audio_dir = '../features/test/'
    # I think weights is the checkpoint and model is the final version
    model_path = '../results/MT_test/model.keras'
    mu = np.load("../results/MT_test/mu.npy")  
    std = np.load("../results/MT_test/std.npy")

    # Experiments
    # experiments(model_path, file_audio_dir, mu, std)

    feature_names = ["vocalic", "consonantal"]  
    posteriorgram = get_posteriorgram_custom(file_audio_feat, model_path, mu, std, feature_names)
    # print(posteriorgram.shape)

    # wav_path = "../test_data/audio_same/071005a_2.wav"
    # plot_features_over_audio(wav_path, posteriorgram, feature_names=feature_names, title="Posteriorgram Features over Audio")

    # TODO Get the actual correct representation.
    # Make sure getting rid of the 40 is correct and that we understand why the model is giving the numbers it is giving.
    # Why does the diagonal seem to always equal 1? This is so strange

