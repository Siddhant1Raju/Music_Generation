# %%
import os
import numpy as np
from pretty_midi import PrettyMIDI
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint

def parse_midi_files(data_directory):
    notes = []
    for file in os.listdir(data_directory):
        if file.endswith(".mid"):
            midi_data = PrettyMIDI(os.path.join(data_directory, file))
            for instrument in midi_data.instruments:
                for note in instrument.notes:
                    notes.append(note.pitch)
    return notes


# %%
def create_sequences(notes, sequence_length):
    note_to_int, int_to_note = create_note_to_int_mapping(notes)

    network_input = []
    network_output = []

    for i in range(len(notes) - sequence_length):
        sequence_in = [note_to_int[note] for note in notes[i:i + sequence_length]]
        sequence_out = note_to_int[notes[i + sequence_length]]
        network_input.append(sequence_in)
        network_output.append(sequence_out)

    # Convert the lists to NumPy arrays and reshape the input data
    network_input = np.array(network_input)
    network_input = np.reshape(network_input, (network_input.shape[0], network_input.shape[1], 1))

    return np.array(network_input), np.array(network_output), note_to_int, int_to_note



# %%
def create_note_to_int_mapping(notes):
    note_to_int = {}
    int_to_note = {}

    unique_notes = set(notes)

    for i, note in enumerate(sorted(unique_notes)):
        note_to_int[note] = i
        int_to_note[i] = note

    return note_to_int, int_to_note

# %%
def train_lstm_model(network_input, network_output, notes):
    model = Sequential()
    model.add(LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(256))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(len(set(notes)), activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=50, batch_size=64, callbacks=callbacks_list)

    return model

# %%
def save_model_and_mapping(model, note_to_int, int_to_note):
    model.save('lstm_model.h5')
    np.save('note_to_int.npy', note_to_int)
    np.save('int_to_note.npy', int_to_note)

# %%
def generate_music(model, network_input, note_to_int, int_to_note, notes, temperature=0.5, sequence_length=100):
    start = np.random.randint(0, len(network_input) - 1)
    pattern = network_input[start].squeeze().tolist()
    pattern = np.array(pattern).reshape(-1, 1)  # Ensure the pattern shape is (100, 1)

    output_notes = []

    for i in range(sequence_length):
        print(f"Pattern shape: {pattern.shape}")
        prediction_input = np.reshape(pattern, (1, sequence_length, 1))  # Correct reshaping
        prediction = model.predict(prediction_input, verbose=0)

        index = sample_next_note(prediction[0], temperature)
        result = int_to_note[index]
        output_notes.append(result)

        pattern = np.roll(pattern, -1)  # Shift pattern elements to the left by one position
        pattern[-1] = index  # Replace the last element with the new index
        pattern = pattern.reshape(-1, 1)  # Ensure the pattern shape is (100, 1)

    return output_notes


# %%
def sample_next_note(predictions, temperature):
    predictions = np.log(predictions) / temperature
    exp_predictions = np.exp(predictions)
    predictions = exp_predictions / np.sum(exp_predictions)

    epsilon = 1e-8
    predictions = (predictions * (1 - epsilon)) + epsilon / len(predictions)
    predictions = np.asarray(predictions).astype('float64')
    predictions = predictions / np.sum(predictions)

    probabilities = np.random.multinomial(1, predictions, 1)
    return np.argmax(probabilities)




# %%
def create_midi_file(prediction_output, int_to_note):
    midi_data = pretty_midi.PrettyMIDI()

    # Create a piano instrument with program number 0 (Acoustic Grand Piano)
    piano_program = 0  
    piano = pretty_midi.Instrument(program=piano_program)
    midi_data.instruments.append(piano)

    last_time = 0
    for note_value in prediction_output:
        duration = np.random.uniform(0.5, 1.0)  # Random duration for each note
        note_pitch = int_to_note[note_value]
        note_velocity = 100  # Velocity set to 100
        piano.notes.append(pretty_midi.Note(
            velocity=note_velocity, pitch=note_pitch, start=last_time, end=last_time + duration))
        last_time += duration

    midi_data.write('generated_music.mid')



# %%
def main():
    # Load and parse MIDI files
    data_directory = "C:/Users/siddh/Desktop/music/midi_songs"
    notes = parse_midi_files(data_directory)

    # Define sequence length and create input-output sequences
    sequence_length = 100
    network_input, network_output, note_to_int, int_to_note = create_sequences(notes, sequence_length)

    # One-hot encode the output
    network_output_one_hot = np.zeros((network_output.shape[0], len(set(notes))))
    network_output_one_hot[np.arange(network_output.shape[0]), network_output] = 1

    # Train LSTM model
    model = train_lstm_model(network_input, network_output_one_hot, notes)

    # Save the trained model and note-to-int mapping
    save_model_and_mapping(model, note_to_int, int_to_note)

    # Generate new music sequences
    prediction_output = generate_music(model, network_input, note_to_int, int_to_note, notes)

    # Convert output sequences to MIDI format and save
    create_midi_file(prediction_output, int_to_note)

if __name__ == "__main__":
    main()



