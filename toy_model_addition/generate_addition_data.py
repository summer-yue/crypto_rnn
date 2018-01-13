import numpy as np

class CharacterTable(object):
    """Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their char output
    """
    def __init__(self, chars):
        """Initialize character table.
        # Arguments
            chars: Characters that can appear in the input.
        """
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        """One hot encode given string C.
        # Arguments
            num_rows: Just number of characters in input C. Number of rows in the returned one hot encoding. This is
                used to keep the # of rows for each data the same.
        """
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in x)

class AdditionDataGenerator(object):
    """ Given the max input digit number, number of data we want to generate and an alphabet,
    + Generate a set number of data points split into training and validation set
    """
    def __init__(self, max_input_digits, data_count, alphabet):
        self.max_input_digits = max_input_digits
        self.data_count = data_count
        self.alphabet = alphabet
        self.table = CharacterTable(alphabet)

    def generate(self):
        """ Generate a set number of data points split into training and validation set
        Returns:
            the two sets of training data with their corresponding labels
        """
        questions = []
        answers = []
        max_input_len = self.max_input_digits * 2 + 1
        max_output_len = self.max_input_digits + 1

        print('Generating data for addition with maximum input digits number:', self.max_input_digits)
        while len(questions) < self.data_count:
            f_generate_random_number = lambda: int(''.join(np.random.choice(list('0123456789'))
                            for i in range(np.random.randint(1, self.max_input_digits + 1))))
            a, b = f_generate_random_number(), f_generate_random_number()

            # Pad the input and output data with spaces so all data have the same length
            q = '{}+{}'.format(a, b)
            q = q + ' ' * (max_input_len - len(q))
            ans = str(a + b)
            ans += ' ' * (max_output_len - len(ans))
            questions.append(q)
            answers.append(ans)

        # Vectorize
        x = np.zeros((len(questions), max_input_len, len(self.alphabet)), dtype=np.bool)
        y = np.zeros((len(questions), max_output_len, len(self.alphabet)), dtype=np.bool)
        for i, q in enumerate(questions):
            x[i] = self.table.encode(q, max_input_len)
        for i, sentence in enumerate(answers):
            y[i] = self.table.encode(sentence, max_output_len)

        #Shuffle x and y
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        x = x[indices]
        y = y[indices]

        #Split into training set and validation set
        split_at = len(x) - len(x) // 10
        (self.x_train, self.x_vali) = x[:split_at], x[split_at:]
        (self.y_train, self.y_vali) = y[:split_at], y[split_at:]

        return self.x_train, self.y_train, self.x_vali, self.y_vali

