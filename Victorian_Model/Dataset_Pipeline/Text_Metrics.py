from collections import Counter
import string

# Function to read text from a plain text file
def read_text(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    return text

# Function to calculate word frequencies
def word_frequency_from_text(file_path):
    # Read text from the text file
    text = read_text(file_path)
    
    # Convert to lowercase to make the count case-insensitive
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Split the text into words
    words = text.split()
    
    # Create a dictionary with word frequencies using Counter
    word_freq = Counter(words)
    
    return word_freq

# Path to your text file
file_path = r"C:\Users\James\Documents\Career\CVs\Data Analyst\James Sharpe CV_Data Analyst No Profile.txt"

# Get the word frequency dictionary
word_frequencies = word_frequency_from_text(file_path)

# Display the word frequency dictionary
print(word_frequencies)
