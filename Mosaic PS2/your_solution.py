from collections import defaultdict
import random
import numpy as np
import string

words = "C:\\Users\\Rohan Sharma\\Desktop\\Mosaic-24-main\\Mosaic PS2\\training.txt"

with open(words, 'r') as file:
    corpus = file.readlines()

np.random.seed(1)

# Developing unigram counts: 
unigram_counts = None

from collections import Counter

# Build unigram model. Count character frequency.
def unigram(corpus):
    unigram_counts = Counter()

    for word in corpus:
        for char in word:
            unigram_counts[char] += 1

    return unigram_counts

unigram_counts = unigram(corpus)
#print(unigram_counts)


unigram_counts_by_length = None

from collections import defaultdict, Counter


# Developing the bigram model:
bigram_counts = None

# add $ to the front of a word since this is a Bigram Model.
def convert_word(word):
    return "$" + word

# Collect bigram counts. Because we don't delete keys in dictionary and counter will return 0 for unseen pattern,
# we don't need to add char for dictionary as Unigram Based on Word Length approach.
def bigram(corpus):
    bigram_counts = defaultdict(Counter)

    for word in corpus:
        word = convert_word(word)

        # generate a list of bigrams
        bigram_list = zip(word[:-1], word[1:])

        # iterate over bigrams
        for bigram in bigram_list:
            first, second = bigram
            bigram_counts[first][second] += 1
    return bigram_counts

bigram_counts = bigram(corpus)
# print(bigram_counts)


# Calculate bigram probability
def bigram_prob(key, char, bigram_counts):
    prev_word_counts = bigram_counts[key]
    total_counts = float(sum(prev_word_counts.values()))

    if total_counts!=0:
        return prev_word_counts[char] / float(sum(prev_word_counts.values()))
    else:
        return 0


def bigram_guesser(mask, guessed, bigram_counts=bigram_counts, unigram_counts=unigram_counts): # add extra arguments if needed

    # available is a list that does not contain the character in guessed
    available = list(set(string.ascii_lowercase) - set(guessed))

    # The probabilities of available character
    bigram_probs = []

    for char in available:
        char_prob = 0
        for index in range(len(mask)):
            # The first case is that the first char has not been guessed
            if index == 0 and mask[index] == '_':
                char_prob +=  bigram_prob('$', char, bigram_counts)

            # The second case is that the other chars have not been guessed
            elif mask[index] == '_':
                # if the previous word has been guessed apply bigram
                if not mask[index - 1] == '_':
                    char_prob +=  bigram_prob(mask[index - 1], char, bigram_counts)

                # If the previous word has not been guessed apply unigram
                else:
                    char_prob +=  unigram_counts[char] / float(sum(unigram_counts.values()))

            # The final case is that the character is guessed so we skip this position
            else:
                continue

        bigram_probs.append(char_prob)

    # Return the max probability of char
    return available[bigram_probs.index(max(bigram_probs))]


# add $$ to the front of a word
def trigram_convert_word(word):
    return "$$" + word

# collect trigram counts
def trigram(corpus):
    trigram_counts = Counter()
    bigram_counts = defaultdict(Counter)

    for word in corpus:
        word = trigram_convert_word(word)

        # generate a list of trigrams
        trigram_list = zip(word[:-2], word[1:-1], word[2:])

        # generate a list of bigrams
        bigram_list = zip(word[:-1], word[1:])

        # iterate over trigrams
        for trigram in trigram_list:
            first, second, third = trigram
            element = first+second+third
            trigram_counts[element] += 1

        # iterate over bigrams
        for bigram in bigram_list:
            first, second = bigram
            bigram_counts[first][second] += 1

    return trigram_counts, bigram_counts

trigram_counts, bigram_counts_for_trigram = trigram(corpus)
# print(trigram_counts)

# Calculate trigram probability
def trigram_prob(wi_2, wi_1, char, trigram_counts, bigram_counts):
    trigram_key = wi_2 + wi_1 + char
    bigram_key = wi_2 + wi_1

    # Check if the bigram exists in the counts
    if bigram_key in bigram_counts and bigram_counts[bigram_key] != 0:
        # Check if the trigram exists in the counts
        if trigram_key in trigram_counts:
            # Return the probability
            return trigram_counts[trigram_key] / float(bigram_counts[bigram_key])
    
    # Return 0.0 if either bigram or trigram count is missing or the bigram count is zero
    return 0.0

    


def trigram_guesser(mask, guessed, bigram_counts=bigram_counts_for_trigram, trigram_counts=trigram_counts,
                          unigram_counts=unigram_counts):

    # available is a list that does not contain the character in guessed
    available = list(set(string.ascii_lowercase) - set(guessed))

    # The probabilities of available character
    trigram_probs = []

    # if len(mask) = 1, means that there is only a character. Therefore, need to pad in order to avoid error from
    # traverse mask[index -2] and mask[index -1]
    mask = ['$', '$'] + list(mask)

    trigram_lambda = 0.45
    bigram_lambda = 0.45
    unigram_lambda = 0.1

    for char in available:
        char_prob = 0
        for index in range(len(mask)):
            # The first case is that the first char has not been guessed
            if index == 0 and mask[index] == '_':
                char_prob += trigram_lambda * trigram_prob('$', '$', char, trigram_counts, bigram_counts)

            # The second case is that the second char has not been guessed
            if index == 1 and mask[index] == '_':
                # If the previous word has been guessed, apply trigram
                if not mask[index - 1] == '_':
                    char_prob += trigram_lambda * trigram_prob('$', mask[index - 1], char, trigram_counts, bigram_counts)

                # If the previous word has not been guessed, apply unigram
                else:
                    char_prob +=  unigram_lambda * unigram_counts[char] / float(sum(unigram_counts.values()))

            # The third case is that the other chars have not been guessed
            elif mask[index] == '_':
                # If wi-2 and wi-1 have been guessed, apply trigram
                if not mask[index - 2] == '_' and not mask[index - 1] == '_':
                    char_prob += trigram_lambda * trigram_prob(mask[index - 2], mask[index - 1], char,
                                                            trigram_counts, bigram_counts)

                # If wi-2 hasn't been guessed but wi-1 has been guessed, apply bigram
                elif mask[index - 2] == '_' and not mask[index - 1] == '_':
                    char_prob += bigram_lambda * bigram_prob(mask[index - 1], char, bigram_counts)

                # If wi-1 hasn't been guessed, apply unigram
                else:
                    char_prob +=  unigram_lambda * unigram_counts[char] / float(sum(unigram_counts.values()))

            # The final case is that the character is guessed so we skip this position
            else:
                continue

        trigram_probs.append(char_prob)

    # Return the max probability of char
    return available[trigram_probs.index(max(trigram_probs))]


# Add $$$ to the front of a word for fourgram conversion
def fourgram_convert_word(word):
    return "$$$" + word



# Collect fourgram counts
def fourgram(corpus):
    fourgram_counts = Counter()
    trigram_counts = defaultdict(Counter)
    bigram_counts = defaultdict(Counter)

    for word in corpus:
        word = fourgram_convert_word(word)

        # Generate a list of fourgrams
        fourgram_list = zip(word[:-3], word[1:-2], word[2:-1], word[3:])

        # Generate a list of trigrams
        trigram_list = zip(word[:-2], word[1:-1], word[2:])

        # Generate a list of bigrams
        bigram_list = zip(word[:-1], word[1:])

        # Iterate over fourgrams
        for fourgram in fourgram_list:
            first, second, third, fourth = fourgram
            element = first + second + third + fourth
            fourgram_counts[element] += 1

        # Iterate over trigrams
        for trigram in trigram_list:
            first, second, third = trigram
            trigram_counts[first + second][third] += 1

        # Iterate over bigrams
        for bigram in bigram_list:
            first, second = bigram
            bigram_counts[first][second] += 1

    return fourgram_counts, trigram_counts, bigram_counts

fourgram_counts, trigram_counts_for_fourgram, bigram_counts_for_fourgram = fourgram(corpus)


# Calculate fourgram probability
def fourgram_prob(wi_3, wi_2, wi_1, char, fourgram_counts=fourgram_counts, trigram_counts=trigram_counts, bigram_counts=bigram_counts):
    if trigram_counts.get(wi_3 + wi_2, {}).get(wi_1, 0) != 0:
        return fourgram_counts.get(wi_3 + wi_2 + wi_1 + char, 0) / float(trigram_counts[wi_3 + wi_2][wi_1])
    else:
        return 0


# Guess a character using the fourgram model
def fourgram_guesser(mask, guessed, trigram_counts=trigram_counts_for_fourgram, fourgram_counts=fourgram_counts):
    available = list(set(string.ascii_lowercase) - set(guessed))
    fourgram_probs = []

    # Pad the mask to avoid errors
    mask = ['$', '$', '$'] + list(mask)

    fourgram_lambda = 0.5
    trigram_lambda = 0.3
    bigram_lambda = 0.15
    unigram_lambda = 0.05

    for char in available:
        char_prob = 0
        for index in range(len(mask)):
            if index == 0 and mask[index] == '_':
                char_prob += fourgram_lambda * fourgram_prob('$', '$', '$', char, fourgram_counts, trigram_counts)

            if index == 1 and mask[index] == '_':
                if not mask[index - 1] == '_':
                    char_prob += fourgram_lambda * fourgram_prob('$', '$', mask[index - 1], char, fourgram_counts, trigram_counts)
                else:
                    char_prob += trigram_lambda * trigram_prob('$', '$', char, trigram_counts, bigram_counts)

            if index == 2 and mask[index] == '_':
                if not mask[index - 1] == '_' and not mask[index - 2] == '_':
                    char_prob += fourgram_lambda * fourgram_prob('$', mask[index - 2], mask[index - 1], char, fourgram_counts, trigram_counts)
                elif not mask[index - 1] == '_':
                    char_prob += trigram_lambda * trigram_prob('$', mask[index - 1], char, trigram_counts, bigram_counts)
                else:
                    char_prob += bigram_lambda * bigram_prob('$', char, bigram_counts)

            elif mask[index] == '_':
                if not mask[index - 1] == '_' and not mask[index - 2] == '_' and not mask[index - 3] == '_':
                    char_prob += fourgram_lambda * fourgram_prob(mask[index - 3], mask[index - 2], mask[index - 1], char, fourgram_counts, trigram_counts)
                elif not mask[index - 2] == '_' and not mask[index - 1] == '_':
                    char_prob += trigram_lambda * trigram_prob(mask[index - 2], mask[index - 1], char, trigram_counts, bigram_counts)
                elif not mask[index - 1] == '_':
                    char_prob += bigram_lambda * bigram_prob(mask[index - 1], char, bigram_counts)
                else:
                    char_prob += unigram_lambda * unigram_counts[char] / float(sum(unigram_counts.values()))

            else:
                continue

        fourgram_probs.append(char_prob)

    return available[fourgram_probs.index(max(fourgram_probs))]



def suggest_next_letter_sol(displayed_word, guessed_letters):
    """_summary_

    This function takes in the current state of the game and returns the next letter to be guessed.
    displayed_word: str: The word being guessed, with underscores for unguessed letters.
    guessed_letters: list: A list of the letters that have been guessed so far.
    Use python hangman.py to check your implementation.
    """
    

    return fourgram_guesser(displayed_word, guessed_letters)

    raise NotImplementedError
