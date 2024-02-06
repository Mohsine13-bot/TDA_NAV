import numpy as np
from tqdm.auto import tqdm
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

# Not a perfect transition matrix because I do not obtain a sum of 1 in bith lines and columns, but works for what I want to do.
def generate_transition_matrix(themes, variability=0.2):

    """
        This method generate a transition matrix using a list of elements and a variability paramteter

        Inputs:

            themes: List of elements (in our case list of themes)

            variability: float to control the probability distribution of the matrix, a lower value give high probability in the diagonal and vise versa

        Output:

            Transition Matrix of shape (len(themes), len(themes))

    """
    #size of the transition matrix
    size = len(themes)

    #Create a distribution with a probability of 1 to stay in the same state and add a drift
    transition_matrix = np.identity(size) + np.random.uniform(low=0, high=variability, size=(size, size))

    #Normalize to get a sum of 1 for each line
    transition_matrix = transition_matrix / np.sum(transition_matrix, axis=1, keepdims=True)

    return transition_matrix


def generate_dirichlet(size, alpha):

    """
        Generate a list of proabilities where the sum is equal to 1, control the entropy using the alpha parameter,higher alpha for a more uniform probability.
        
            Inputs:

                size: (int) The size of the list

                alpha: (float) control the entropy of the output

            Ouput:

                List of probabilities with the specified size

    """

    return np.random.dirichlet(np.ones(size)*alpha, size=1)[0]


def generate_themes_documents_probabilities(data, alpha):

    """
        Generate a list of probabilities of the documents picking for each theme.

            Inputs:

                data: A dictionary that contains the themes names and documents titles

                alpha: alpha: (float) control the entropy of the output

                
            Outpus:

                Lists of probabilities for each themes


    """

    result = []

    for theme in data:

        result.append(generate_dirichlet(len(data[theme]), alpha))

    return result


#I know it's not perfect because usually when 
def generate_session(data, themes_transitions, documents_probabilities, session_len):

    """

        Generate fake session user using a theme transition matrix, a document picking probability and a session length

            Inputs:

                data: A dictionary that contains the themes names and documents titles

                themes_transitions: 2D array transition matrix that contains the probabilities to go from one thme to another

                docuements_probabilities: 2D array that contains the probabilities of picking a document

                session_len: (int) The size of the generated sequence

            Output:

                List of couples (theme, document)
                

    """

    session  = []

    themes = list(data.keys())

    #Select a starting theme randomly (uniform probability)
    theme_id = np.random.choice(np.arange(themes_transitions.shape[0]))

    selected_theme = themes[theme_id]

    #Select the first docuement and add it to the list
    document = np.random.choice(data[selected_theme], p=documents_probabilities[theme_id])

    session.append((selected_theme, document))

    for _ in range(session_len-1):

        theme_id = np.random.choice(np.arange(themes_transitions.shape[0]), p=themes_transitions[theme_id])

        selected_theme = themes[theme_id]

        #Select the first docuement and add it to the list
        document = np.random.choice(data[selected_theme], p=documents_probabilities[theme_id])

        session.append((selected_theme, document))


    return session

# To generate sessions of the same behaviour
def generate_sessions(data, themes_transitions, documents_probabilities, n_sessions, length_category='medium'):

    """
        Generate a list of sessions that have the same length category

            Inputs:

                data: A dictionary that contains the themes names and documents titles

                themes_transitions: 2D array transition matrix that contains the probabilities to go from one thme to another

                docuements_probabilities: 2D array that contains the probabilities of picking a document

                n_sessions: (int) The number of sessions to generate

                length_category: (str) The size category of session "small" (sequence of 1 to 2), "medium" (3 to 5) or "long" (5 to 20)
            
                
            Output:

                List of sessions
    """

    categories = ["small", "medium", "long"]

    assert length_category in categories, "Unknown specified length category please choose between small, medium or long"

    if length_category == categories[0]:

        sequences_sizes = [1, 2]

        return [generate_session(data, themes_transitions, documents_probabilities, np.random.choice(sequences_sizes)) for _ in range(n_sessions)]
    
    elif length_category == categories[1]:

        sequences_sizes = np.arange(start=3, stop=6)

        return [generate_session(data, themes_transitions, documents_probabilities, np.random.choice(sequences_sizes)) for _ in range(n_sessions)]
    
    else:

        sequences_sizes = np.arange(start=5, stop=20)

        return [generate_session(data, themes_transitions, documents_probabilities, np.random.choice(sequences_sizes)) for _ in range(n_sessions)]



def generate_dataset(behaviours_specify, data):

    """
        Generate a dataset of diffrent sessions depending on the list of sepecfied behaviours

            Inputs:

                behaviours_specify: List of parameters to construct sessions to respect a behaviour. (variability, alpha, n_session, length category)

                data: data: A dictionary that contains the themes names and documents titles

            Output:

                List of diffrent sessions of diffrent behaviours


    """

    result = []

    #Generate transitions matrices and their associated documents probabilities
    themes_transitions_matrices = [generate_transition_matrix(data.keys(), variability=b[0]) for b in behaviours_specify]

    documents_prob_pick = [generate_themes_documents_probabilities(data, alpha=b[1]) for b in behaviours_specify]

    for i, b in enumerate(tqdm(behaviours_specify)):
        
        result += generate_sessions(data, themes_transitions_matrices[i], documents_prob_pick[i], b[2], b[3])

    return result



def remove_stop_words(sentence):

    """
        Remove stop words and ponctuation in a sentence and return a filtred list of words

            Input:

                sentence: (str)

            Output:

                A list of filtred words
    """

    cleaned_text = re.sub(r'[^\w\s]|l\'', '', sentence)

    words = word_tokenize(cleaned_text)

    filtered_words = [word.lower() for word in words if word.lower() not in stopwords.words('french')]

    return filtered_words


def get_document_words(document):

    """
        Get a list of words associated to a document (theme + words in title)

        Input:

            document: (dict) It contains the title and the theme

        Output:

            A filtred list of words


    """

    representation = [document[0].lower()]

    representation += remove_stop_words(document[1]["title"])

    return representation



def get_session_words(session):

    """

        Get a set of words in a specific session

            Input:

                session: List of dictionaries that contains the themes and the titles


            Output:

                A set of filtred words
    
    """

    result = []

    for document in session:

        result += get_document_words(document)

    return set(result)


def get_session_representation(session, model):

    """

        Get a representation of a session based on the words in the title and the visited themes

            Inputs:

                session: List of documments that have titles and themes

                model: The word2vec model to get representation of words

                
            Output:

                Numpy array of the representations
    
    """

    session_words = get_session_words(session)

    return np.array([model[word] for word in session_words if word in model.key_to_index]).T



def get_svd_session_representation(session_representation):

    """
        Get the first eigen vector of the matrix session representation

            Input:

                session_representation: 2D Matrix produced by the egt_session_representation method

            Output:

                A 1D numpy array (first eigen vector)
    """

    return np.linalg.svd(session_representation)[0][0,:]


def get_sessions_representation(sessions, model):

    """

        Return the first eigen vector for each session representation matrix

            Input:

                sessions: List of sessions (one session is alist of picked documents)

                model: The word2vec model to get the representation of each word in the set of words in session
                
            Output:

                A list of vectors each vector represent a session
    
    """

    print("Vectorize sessions...")

    sessions_word2vec = [get_session_representation(session, model) for session in tqdm(sessions)]

    print("Done!\nCompute SVD sessions")

    sessions_reduced = [get_svd_session_representation(session) for session in tqdm(sessions_word2vec)]

    print("Done!")

    return sessions_reduced


