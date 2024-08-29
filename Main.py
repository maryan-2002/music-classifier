import pandas as pd
# from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import ttk
import csv
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from PIL import Image, ImageTk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier

# Create the main window
window = tk.Tk()
window.title("Music Classifier")
window.geometry("500x850")  # Set the desired dimensions

# Load and display the image
background_image_path = "C:\\Users\\Asus\\Desktop\\4th year\\4th year 1st sem-2024\\AI\\AIPROJJ\\pythonProject\\music.jpeg"
background_image = Image.open(background_image_path)
background_image = ImageTk.PhotoImage(background_image)

image_label = ttk.Label(window, image=background_image)
image_label.grid(row=1, column=0, padx=0, pady=0)

# Title Label
title_label = ttk.Label(window, text="Music Classification model", font=("Georgia", 25, "bold"),
                        foreground="darkorange", background='navy')

title_label.grid(row=0, column=0, columnspan=2, pady=(20, 10))

subtitle_label = ttk.Label(window, text="Welcome to your Ai model!", font=("Times New Roman", 14), foreground="black",
                           background='orangered')
subtitle_label.grid(row=2, column=0, columnspan=1, pady=(0, 20))

subtitle_label = ttk.Label(window, text="Performance Evaluation For the Decision Tree Model",
                           font=("Times New Roman", 15), background="darkorange", foreground="navy")
subtitle_label.grid(row=3, column=0, columnspan=1, pady=(0, 20))

###############################################

# Step 1: Load and Prepare the Training Data
training_data_path = "C:\\Users\\Asus\\Desktop\\4th year\\4th year 1st sem-2024\\AI\\AIPROJJ\\pythonProject\\Training data.csv"
Training_data = pd.read_csv(training_data_path)

print(Training_data.shape)  # to make sure that all data are in df
features = Training_data[['artist', 'song', 'Type', 'Release Year', 'listener-Gender', 'listener-Age']]
labels = Training_data['likedSong']  # Assuming 'likedSong' is your target variable

# One-hot encoding for categorical variables
categorical_cols = ['artist', 'song', 'Type', 'listener-Gender']
features_encoded = pd.get_dummies(Training_data[categorical_cols], columns=categorical_cols)

# Handle 'listener-Age' with ordinal encoding
age_mapping = {'20\'s': 1, '30\'s': 2, '40\'s,50\'s': 3}  # Add more mappings as needed
features_encoded['listener-Age'] = Training_data['listener-Age'].map(age_mapping)
gender_mapping = {'F': 1, 'M': 2}

# Include 'Release Year' if it's numerical
features_encoded['Release Year'] = Training_data['Release Year']

###############################################

# Step 2: Train the Decision Tree Model
X_train, X_val, y_train, y_val = train_test_split(features_encoded, labels, test_size=0.3, random_state=42)
# Create and train the decision tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)


# Make predictions on the validation set

# Make testing on the validation set
def prediction(X_test, model):
    y_pred = model.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    # Print relevant information for samples with 'Yes' class
    return y_pred


###############################################

#  function for cal_accuracy
def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix for decision tree: ", confusion_matrix(y_test, y_pred))
    print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
    subtitle_label = ttk.Label(window, text=("Accuracy =", accuracy_score(y_test, y_pred) * 100),
                               font=("Times New Roman", 12), background="darkblue", foreground="white")
    subtitle_label.grid(row=6, column=0, columnspan=1, pady=(0, 20))


# Function to calculate recall

def cal_recall(y_test, y_pred):
    recall_macro = recall_score(y_test, y_pred, average='macro')
    recall_micro = recall_score(y_test, y_pred, average='micro')
    subtitle_label = ttk.Label(window, text=("Recall  =", recall_macro * 100), font=("Times New Roman", 12),
                               background="darkblue", foreground="white")
    subtitle_label.grid(row=5, column=0, columnspan=1, pady=(0, 20))
    print("Recall : ", recall_macro * 100)


# Function to calculate precision
from sklearn.metrics import precision_score


# Function to calculate precision
def cal_precision(y_test, y_pred):
    precision = precision_score(y_test, y_pred, average='weighted')
    precision_percent = precision * 100
    subtitle_label = ttk.Label(window, text=("Precision =", precision_percent), font=("Times New Roman", 12),
                               background="darkblue", foreground="white")
    subtitle_label.grid(row=4, column=0, columnspan=1, pady=(0, 20))
    print("Precision: ", precision_percent)


###############################################

# Make predictions on the validation set
predictions = prediction(X_val, model)

# Call the cal_accuracy function
cal_accuracy(y_val, predictions)
# Call the cal_recall function
cal_recall(y_val, predictions)
cal_precision(y_val, predictions)

# Printing names of songs that the user liked
liked_song_indices = [i for i, pred in enumerate(predictions) if pred == 'Yes']
liked_song_names = features.loc[y_val.index[liked_song_indices], 'song'].tolist()

print("Songs liked by the user:")
for song in liked_song_names:
    print(song)

# Displaying names of songs that the user liked
liked_songs_label = ttk.Label(window, text="Songs liked by the user:", font=("Times New Roman", 14),
                              background="darkorange", foreground="black")
liked_songs_label.grid(row=7, column=0, columnspan=1, pady=(0, 10))

liked_songs_combobox = ttk.Combobox(window, values=liked_song_names, font=("Times New Roman", 14), state="readonly")
liked_songs_combobox.grid(row=8, column=0, columnspan=1, pady=(0, 10))
liked_songs_combobox.current(0)

subtitle_label = ttk.Label(window, text="Performance Evaluation For the Neural Network Model",
                           font=("Times New Roman", 15), background="darkorange", foreground="navy")
subtitle_label.grid(row=9, column=0, columnspan=1, pady=(0, 20))

###############################################

# Load the data
file_path2 = "C:\\Users\\Asus\\Desktop\\4th year\\4th year 1st sem-2024\\AI\\AIPROJJ\\pythonProject\\Training data.csv"
data = pd.read_csv(file_path2, delimiter=',')

# Check the column names and structure of the loaded data
print(data.columns)
print(data.head())

# Define features (X) and target variable (y)
X = data.drop('likedSong', axis=1)
y = data['likedSong']

# Initialize LabelEncoder for categorical columns
label_encoders = {}
categorical_cols = ['artist', 'song', 'Type', 'Release Year', 'listener-Gender', 'listener-Age']

# Encode categorical columns
for column in categorical_cols:
    label_encoders[column] = LabelEncoder()
    X[column] = label_encoders[column].fit_transform(X[column])

###############################################

# Split the data into training and testing sets
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the MLPClassifier (Neural Network)
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)  # Example architecture

# Train the model on the training data
mlp.fit(X_train1, y_train1)

# Predict on the test set
y_pred1 = mlp.predict(X_test1)
print(y_pred1)
print("Confusion Matrix for neural network: ", confusion_matrix(y_test1, y_pred1))

###############################################

# Evaluate the model's performance

accuracy = accuracy_score(y_test1, y_pred1)
print(f"Accuracy: {accuracy * 100.0:.2f}")
subtitle_label = ttk.Label(window, text=("Accuracy =", accuracy_score(y_test1, y_pred1) * 100),
                           font=("Times New Roman", 12), background="darkblue", foreground="white")
subtitle_label.grid(row=12, column=0, columnspan=1, pady=(0, 20))

recall1 = recall_score(y_test1, y_pred1, average='weighted')
print(f"Recall: {recall1 * 100.0:.2f}")
subtitle_label = ttk.Label(window, text=("Recall  =", recall1 * 100.0), font=("Times New Roman", 12),
                           background="darkblue", foreground="white")
subtitle_label.grid(row=10, column=0, columnspan=1, pady=(0, 20))

# Calculate precision
precision1 = precision_score(y_test1, y_pred1, average='weighted')
print(f"Precision: {precision1 * 100.0:.2f}")
subtitle_label = ttk.Label(window, text=("Precision =", precision1 * 100.0), font=("Times New Roman", 12),
                           background="darkblue", foreground="white")
subtitle_label.grid(row=11, column=0, columnspan=1, pady=(0, 20))

###############################################

# Get indices of songs liked (predicted as 'Yes')
liked_indices = [i for i, label in enumerate(y_pred1) if label == 'Yes']

# Extract and print names of songs liked by the user based on predictions
liked_indices = [i for i, label in enumerate(y_pred1) if label == 'Yes']
liked_songs = data.loc[y_test1.index[liked_indices], 'song']
liked_song_list1 = liked_songs.values.tolist()
print("Songs liked by the user according to neural network model:")
for song in liked_song_list1:
    print(song)

# Displaying names of songs that the user liked
liked_songs_label = ttk.Label(window, text="Songs liked by the user:", font=("Times New Roman", 14),
                              background="darkorange", foreground="black")
liked_songs_label.grid(row=13, column=0, columnspan=1, pady=(0, 10))

liked_songs_combobox = ttk.Combobox(window, values=liked_song_list1, font=("Times New Roman", 14), state="readonly")
liked_songs_combobox.grid(row=14, column=0, columnspan=1, pady=(0, 10))
liked_songs_combobox.current(0)
#############################################

#############################################
# Run the main loop
window.configure(bg="darkorange")


def test_by_user_input():
    result_label["text"] = "Test By User Input"


window2 = tk.Tk()
window2.title("Test By User Input Interface")
window2.geometry("500x500")

result_label = ttk.Label(window2, text="Artist Name:")
result_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
entry_artist = ttk.Combobox(window2, values=["ABBA", "Ace Of Base", "Adam Sandler", "Adele", "Aerosmith", "Air Supply",
                                             "Aiza Seguerra", "Alabama", "Alan Parsons Project", "Alice Cooper",
                                             "Alice In Chains", "Alison Krauss", "Allman Brothers Band", "Alphaville",
                                             "America", "Amy Grant", "Andy Williams", "Annie", "Ariana Grande",
                                             "Arlo Guthrie", "Arrogant Worms", "Avril Lavigne", "Backstreet Boys",
                                             "Barbie", "Barbra Streisand", "Beach Boys", "The Beatles",
                                             "Beautiful South", "Bee Gees", "Bette Midler", "Bill Withers",
                                             "Billie Holiday", "Billy Joel", "Bing Crosby", "Black Sabbath", "Blur",
                                             "Bob Dylan", "Bob Dylan", "Bob Seger", "Bon Jovi", "Boney M.",
                                             "Bonnie Raitt", "Bosson", "Bread", "Britney Spears", "Bruce Springsteen",
                                             "Bruno Mars", "Cake", "Carly Simon", "Carol Banawa", "Carpenters",
                                             "Cat Stevens", "Celine Dion", "Chaka Khan"])
entry_artist.grid(row=0, column=1, padx=10, pady=10)

result_label = ttk.Label(window2, text="Song Name:")
result_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")
entry_song = ttk.Entry(window2)
entry_song.grid(row=1, column=1, padx=10, pady=10)

result_label = ttk.Label(window2, text="Type:")
result_label.grid(row=2, column=0, padx=10, pady=10, sticky="w")
entry_type = ttk.Combobox(window2,
                          values=["Romantic", "Rock", "Jazz", "Ballads", "Folk", "Country", "Classical", "Hip Hop"])
entry_type.grid(row=2, column=1, padx=10, pady=10)

result_label = ttk.Label(window2, text="Release Year:")
result_label.grid(row=3, column=0, padx=10, pady=10, sticky="w")
entry_release_year = ttk.Combobox(window2,
                                  values=["1999", "2000", "2002", "2005", "2012", "2014", "2017", "2018", "2019",
                                          "2021", "2020", "2023"])
entry_release_year.grid(row=3, column=1, padx=10, pady=10)

result_label = ttk.Label(window2, text="Gender:")
result_label.grid(row=4, column=0, padx=10, pady=10, sticky="w")
gender_var = tk.StringVar()
entry_gender = ttk.Combobox(window2, textvariable=gender_var, values=["M", "F"])
entry_gender.grid(row=4, column=1, padx=10, pady=10)

result_label = ttk.Label(window2, text="Age:")
result_label.grid(row=5, column=0, padx=10, pady=10, sticky="w")
entry_age = ttk.Combobox(window2, values=["20\'s", "30\'s", "40\'s  ,50\'s"])
entry_age.grid(row=5, column=1, padx=10, pady=10)

# Create a label for displaying the prediction result
result_display_label = ttk.Label(window2, text="")
result_display_label.grid(row=7, column=0, columnspan=2, pady=10)

def test_by_user_input():
    # Retrieve user input
    artist = entry_artist.get()
    song_name = entry_song.get()
    song_type = entry_type.get()
    release_year = int(entry_release_year.get())  # Assuming release year is a numerical field
    gender = entry_gender.get()
    age = entry_age.get()

    # Create a DataFrame with user input
    user_input_data = pd.DataFrame({
        'artist': [artist],
        'song': [song_name],
        'Type': [song_type],
        'Release Year': [release_year],
        'listener-Gender': [gender],
        'listener-Age': [age]
    })

    # Perform one-hot encoding and ordinal encoding on user input
    user_input_encoded = pd.get_dummies(user_input_data)

    # Create a DataFrame for missing columns with default value of 0
    missing_columns = {col: [0] for col in features_encoded.columns if col not in user_input_encoded.columns}
    missing_df = pd.DataFrame(missing_columns)


    user_input_encoded = pd.concat([user_input_encoded, missing_df], axis=1)

    # Reorder columns to match the training set
    user_input_encoded = user_input_encoded[features_encoded.columns]



    # Make prediction using the trained Decision Tree Model
    user_prediction = model.predict(user_input_encoded)

    # Display the prediction result in the result_display_label
    result_display_label["text"] = f"Prediction: {user_prediction[0]}"


test_button = ttk.Button(window2, text="Submit", command=test_by_user_input)
test_button.grid(row=6, column=0, columnspan=2, pady=20)

# Display the prediction result

# ... (أي عناصر أخرى في واجهة الاختبار الثانية)
window.mainloop()
window2.mainloop()





