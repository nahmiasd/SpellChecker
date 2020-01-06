import numpy as np
from random import randint
import operator
import itertools

def learn_language_model(files, n=3, lm=None):
    """ Returns a nested dictionary of the language model based on the
    specified files.Text normalization is expected (please explain your choice
    of normalization HERE in the function docstring.
    Example of the returned dictionary for the text 'w1 w2 w3 w1 w4' with a
    tri-gram model:
    tri-grams:
    <> <> w1
    <> w1 w2
    w1 w2 w3
    w2 w3 w1
    w3 w1 w4
    w1 w4 <>
    w4 <> <>
    and returned language model is:
    {
    w1: {'':1, 'w2 w3':1},
    w2: {w1:1},
    w3: {'w1 w2':1},
    w4:{'w3 w1':1},
    '': {'w1 w4':1, 'w4':1}
    }

    Args:
          files (list): a list of files (full path) to process.
          n (int): length of ngram, default 3.
          lm (dict): update and return this dictionary if not None.
                     (default None).

    Returns:
        dict: a nested dict {str:{str:int}} of ngrams and their counts.
    """
    ngrams = {}
    for infile in files:
        with open(infile, 'r') as file:
            readf = file.read()
            lang = ''.join(c for c in readf if c.isalnum() or c == ' ')
            lang = lang.lower().strip()
            words = ['' for i in range(n - 1)]
            words += lang.split()
            words += ['' for i in range(n - 1)]
            for i in range(len(words) - n):
                gram = []
                for g in range(n):
                    gram.append(words[i + g])
                lastword = gram[n - 1]
                if lastword not in ngrams:
                    ngrams[lastword] = {}
                countdict = ngrams[lastword]
                if gram[n - 2] == '':
                    previousWords = ''
                else:
                    previousWords = ' '.join([x for x in gram[:-1] if x != ''])
                if previousWords not in countdict:
                    countdict[previousWords] = 1
                else:
                    countdict[previousWords] += 1
    if lm != None:
        for w in ngrams:
            if w not in lm:
                lm[w] = ngrams[w]
                continue
            for k in ngrams[w]:
                if k not in lm[w]:
                    lm[w][k] = ngrams[w][k]
                else:
                    lm[w][k] += ngrams[w][k]
        return lm
    return ngrams


def create_error_distribution(errors_file, lexicon):
    """ Returns a dictionary {str:dict} where str is in:
    <'deletion', 'insertion', 'transposition', 'substitution'> and the inner dict {tupple: float} represents the confution matrix of the specific errors
    where tupple is (err, corr) and the float is the probability of such an error. Examples of such tupples are ('t', 's'), ('-', 't') and ('ac','ca').
    Notes:
        1. The error distributions could be represented in more efficient ways.
           We ask you to keep it simpel and straight forward for clarity.
        2. Ultimately, one can use only 'deletion' and 'insertion' and have
            'sunstiturion' and 'transposition' derived. Again,  we use all
            four explicitly in order to keep things simple.
    Args:
        errors_file (str): full path to the errors file. File format mathces
                            Wikipedia errors list.
        lexicon (dict): A dictionary of words and their counts derived from
                        the same corpus used to learn the language model.

    Returns:
        A dictionary of error distributions by error type (dict).

    """
    distribution = {}
    distribution["deletion"] = dict()
    distribution["insertion"] = dict()
    distribution["transposition"] = dict()
    distribution["substitution"] = dict()
    errDict = generateErrorsDict(errors_file)
    unigrams = getUnigrams(lexicon)
    bigrams = getBigrams(lexicon)
    bigramsAvg = sum([x for x in bigrams.values()]) / float(len(bigrams.keys()))
    unigramsAvg = sum([x for x in unigrams.values()]) / float(len(unigrams.keys()))
    for err in errDict:  # counting
        # corr=errDict[err]
        for corr in errDict[err]:
            levDistTup = get_ops(corr, err)[0]
            errType = levDistTup[0]
            tup = getErrTuple(err, corr, levDistTup)
            if tup not in distribution[errType].keys():
                distribution[errType][tup] = 1
            else:
                distribution[errType][tup] += 1
    # factoring:
    for tup in distribution["deletion"]:
        if tup in bigrams:
            distribution["deletion"][tup] = float(distribution["deletion"][tup]) / bigrams[tup]
        else:
            distribution["deletion"][tup] = float(distribution["deletion"][tup]) / bigramsAvg
    for tup in distribution["transposition"]:
        if tup in bigrams:
            distribution["transposition"][tup] = float(distribution["transposition"][tup]) / bigrams[tup]
        else:
            distribution["transposition"][tup] = float(distribution["transposition"][tup]) / bigramsAvg
    for tup in distribution["insertion"]:
        if tup in unigrams:
            distribution["insertion"][tup] = float(distribution["insertion"][tup]) / unigrams[tup[0]]
        else:
            distribution["insertion"][tup] = float(distribution["insertion"][tup]) / unigramsAvg
    for tup in distribution["substitution"]:
        if tup in unigrams:
            distribution["substitution"][tup] = float(distribution["substitution"][tup]) / unigrams[tup[1]]
        else:
            distribution["substitution"][tup] = float(distribution["substitution"][tup]) / unigramsAvg
    return distribution


def getErrTuple(err, corr, levDistTup):
    """
    returns a tuple of the edit operation according to the format of the confusion matrix given the output of the levinstein distance algorithm
    Args:
        err (str): error word
        corr (str): correction word
        levDistTup (tuple): the output of the levinstein distance algorithm

    Returns: a tuple of (char,char) representing an input in the confusion matrix

    """
    errType = levDistTup[0]
    i = levDistTup[1]
    j = levDistTup[2]
    if errType == "deletion":
        if levDistTup[2] == 0:
            return ('-', corr[levDistTup[2]])
        if not corr[j].isalnum():
            return (corr[levDistTup[2] - 1], '-')
        return (corr[levDistTup[2] - 1], corr[levDistTup[2]])
    if errType == "insertion":
        if j == 0:
            return ('-', err[j])
        return (corr[i], err[j])
    if errType == "transposition":
        return (corr[j], corr[i])
    return (err[i], corr[i])  # sustitution


def generateErrorsDict(errorfile):
    """
    generates a dictionary mapping an error to it's correction
    Args:
        errorfile (str): location of the errors file

    Returns: {str:list} error->list of possible corrections (dict)

    """
    ans = {}
    with open(errorfile, 'r') as file:
        lst = file.read().lower().splitlines()
        for line in lst:
            splt = line.split("->")
            if ',' in splt[1]:
                ans[splt[0]] = [x.strip() for x in splt[1].split(',')]
            else:
                ans[splt[0]] = [splt[1]]
    return ans


def generateLexicon(infile):
    """
    generates lexicon from input file (words to count dictionary)
    Args:
        infile (str): location of input file

    Returns:
        dictonary of {str:int} mapping each word to her count

    """
    ans = {}
    with open(infile, 'r') as file:
        readf = file.read()
        lang = ''.join(c for c in readf if c.isalnum() or c == ' ').lower().strip()
        splt = lang.split()
        for word in splt:
            if word not in ans:
                ans[word] = 1
            else:
                ans[word] += 1
    return ans


def getBigrams(lexicon):
    """
    returns bigrams of letters for each word in the lexicon
    Args:
        lexicon (dict): input lexicon

    Returns:
        dictionary mapping bigram to it's count {str:int} (dict)

    """
    ans = {}
    for word in lexicon:
        word2 = '-' + word + '-'
        for i in range(len(word2) - 1):
            bigram = (word2[i], word2[i + 1])
            if bigram not in ans:
                ans[bigram] = lexicon[word]
            else:
                ans[bigram] += lexicon[word]
    return ans


def getUnigrams(lexicon):
    """
    returns unigrams of character according to the input lexicon
    Args:
        lexicon (dict): lexicon mapping each word to her count

    Returns:
        dictionary {str:int} mapping each letter to her count (dict)
    """
    ans = {}
    for word in lexicon:
        for char in word:
            if char not in ans:
                ans[char] = lexicon[word]
            else:
                ans[char] += lexicon[word]
    return ans


def generate_text(lm, m=15, w=None):
    """ Returns a text of the specified length, generated according to the
     specified language model using the specified word (if given) as an anchor.

     Args:
        lm (dict): language model used to generate the text.
        m (int): length (num of words) of the text to generate (default 15).
        w (str): a word to start the text with (default None)

    Returns:
        A sequrnce of generated tokens, separated by white spaces (str)
    """
    lword=list(lm.keys())[randint(0,len(lm.keys())-1)]
    senLst=[]
    while len(senLst)<m:
        toadd=max(lm[lword].iteritems(), key=operator.itemgetter(1))[0]
        senLst+=toadd.split()[::-1]
        lword=senLst[-1]
    while(w is not None and w not in senLst):
        dict = lm[lword]
        l = [x for x in dict if w in x.split()]
        if len(l) == 0:
            toadd = list(dict.keys())[randint(0,len(dict.keys())-1)] #choose another random word
        else:
            toadd = l[randint(0, len(l) - 1)] #choose the specified word
        senLst += toadd.split()[::-1]
        lword = senLst[-1]
    senLst=senLst[::-1]
    if w is None:
        return " ".join(senLst[0:m])
    index=senLst.index(w)
    return " ".join(senLst[index:index+m])





def correct_word(w, word_counts, errors_dist):
    """ Returns the most probable correction for the specified word, given the specified prior error distribution.

    Args:
        w (str): a word to correct
        word_counts (dict): a dictionary of {str:count} containing the
                            counts  of uniqie words (from previously loaded
                             corpora).
        errors_dist (dict): a dictionary of {str:dict} representing the error
                            distribution of each error type (as returned by
                            create_error_distribution() ).

    Returns:
        The most probable correction (str).
    """
    candidates=[can for can in word_counts if len(get_ops(can,w))<=2 and abs(len(can)-len(w))<=2 and len(can)>1]
    N=sum([val for val in word_counts.values()])
    prob=-1
    correction=None
    for can in candidates:
        prtc=getErrDistError(can,w,errors_dist)
        prior=(word_counts[can]+0.5)/float(N)
        if prior*prtc>prob:
            prob=prior*prtc
            correction=can
    return correction

def getErrDistError(can,w,errors_dist):
    """
    returns the noisy channel probability of a candidate given the word and the confusion matrices
    Args:
        can (str): candidate word
        w (str): text word
        errors_dist (dict): confusion matrices

    Returns:
        probability (float)
    """
    editops = get_ops(can, w)
    prtc = 1
    for tup in editops:
        try:
            errtup = getErrTuple(w, can, tup)
        except:
            mid = execute_ops(editops[::-1], can, w)[1]
            errtup = getErrTuple(mid, can, tup)
        try:
            prtc *= errors_dist[tup[0]][errtup]
        except:
            prtc *= min(errors_dist[tup[0]].values()) #smoothing
    return prtc


def correct_sentence(s, lm, err_dist, c=2, alpha=0.95):
    """ Returns the most probable sentence given the specified sentence, language
    model, error distributions, maximal number of suumed erroneous tokens and likelihood for non-error.

    Args:
        s (str): the sentence to correct.
        lm (dict): the language model to correct the sentence accordingly.
        err_dist (dict): error distributions according to error types
                        (as returned by create_error_distribution() ).
        c (int): the maximal number of tokens to change in the specified sentence.
                 (default: 2)
        alpha (float): the likelihood of a lexical entry to be the a correct word.
                        (default: 0.95)

    Returns:
        The most probable sentence (str)

    """
    wordsCount=countLanguageModel(lm)
    sentence=s.split()
    corrections=[correct_word(w,wordsCount,err_dist) for w in sentence]
    C=set()
    perms=["".join(seq) for seq in itertools.product("01", repeat=len(sentence))]
    for perm in perms:
        C.add(permute(sentence,corrections,perm))
    C.add(s)
    C=[x for x in C if sentencesDistance(x,s)<=c]
    n=max([len(x.split()) for x in lm.values()[0].keys()])
    n+=1
    prob=0
    answer=None
    for candidate in C:
        pw=evaluate_text(candidate,n,lm)
        pxw=getChannelModelProb(candidate,s,err_dist,alpha)
        if(pw*pxw>prob):
            prob=pw*pxw
            answer=candidate
    return answer




def getChannelModelProb(candidate,sentence,errdist,alpha):
    """
    returns the noisy channel probability of a candidate *sentence* given the original sentence and confusion matrices
    Args:
        candidate (str): candidate sentence
        sentence (str): sentence we wish to correct
        errdist (dict): confusion matrices
        alpha: probability of a word in the original sentence to be correct

    Returns:
        the noisy channel probability of the candidate (float)
    """
    s=sentence.split()
    c=candidate.split()
    prob=1
    for i in range(len(s)):
        if c[i]==s[i]:
            prob*=alpha
        else:
            prob*=getErrDistError(c[i],s[i],errdist)
    return prob





def permute(sentence,corrections,perm):
    """
    A helper function that creates a permutation of the sentence given the original words and candidate words
    Args:
        sentence (list): original sentence
        corrections (list): candidate corrections
        perm (str): binary string representing the premutation (1 represent a correction, 0 an original word)

    Returns:

    """
    lst=[]
    for i in range(len(perm)):
        if perm[i]=='0':
            lst.append(sentence[i])
        else:
            lst.append(corrections[i])
    return " ".join(lst)

def sentencesDistance(sen1,sen2):
    """
    A helper function that counts the number of different words between two sentences
    Args:
        sen1 (str): first sentence
        sen2 (str): second sentence

    Returns: number of different words between sentences (int)

    """
    count=0
    s1=sen1.split()
    s2=sen2.split()
    for i in range(len(s1)):
        if s1[i]!=s2[i]:
            count+=1
    return count

def countLanguageModel(lm):
    """
    A helper functions that map each word in the language model to its count
    Args:
        lm (dict): language model

    Returns: word->count (dict)

    """
    ans=dict()
    for word in lm:
        ngrams=lm[word]
        ans[word]=sum([val for val in ngrams.values()])
    return ans

def evaluate_text(s, n, lm):
    """ Returns the likelihood of the specified sentence to be generated by the
    the specified language model.

    Args:
        s (str): the sentence to evaluate.
        n (int): the length of the n-grams to consider in the language model.
        lm (dict): the language model to evaluate the sentence by.

    Returns:
        The likelihood of the sentence according to the language model (float).
    """
    prob=1
    V=len(lm.keys())
    wordCounts=countLanguageModel(lm)
    N=sum(wordCounts.values())
    lst=s.split()[::-1]
    for i in range(len(lst)-n+1):
        word=lst[i]
        grams=" ".join(lst[i+1:i+n][::-1])
        try:
            if grams in lm[word]:
                prob*=float(lm[word][grams])+1/(wordCounts[word] + V)
            else:
                prob*=float(1)/(wordCounts[word] + V) #laplace
        except:
            prob*=1/float(N+V) #laplace
    return prob


def _levenshtein_distance_matrix(string1, string2, is_damerau=False):
    """
    computes the levenshtein distance between two words
    Args:
        string1 (str): first word
        string2 (str): second word
        is_damerau (bool): states if we should consider transposition

    Returns:
        levinstein distance matrix of two words (np.matrix)

    """
    n1 = len(string1)
    n2 = len(string2)
    d = np.zeros((n1 + 1, n2 + 1), dtype=int)
    for i in range(n1 + 1):
        d[i, 0] = i
    for j in range(n2 + 1):
        d[0, j] = j
    for i in range(n1):
        for j in range(n2):
            if string1[i] == string2[j]:
                cost = 0
            else:
                cost = 1
            d[i + 1, j + 1] = min(d[i, j + 1] + 1,  # insert
                                  d[i + 1, j] + 1,  # delete
                                  d[i, j] + cost)  # replace
            if is_damerau:
                if i > 0 and j > 0 and string1[i] == string2[j - 1] and string1[i - 1] == string2[j]:
                    d[i + 1, j + 1] = min(d[i + 1, j + 1], d[i - 1, j - 1] + cost)  # transpose
    return d


def get_ops(string1, string2, is_damerau=True):
    """
    computes levinstein distance matrix and retruns the edit operations between two words
    Args:
        string1 (str): first word
        string2 (str): second word
        is_damerau (bool): consider transposition

    Returns:
        list of edit operations (list)
    """
    dist_matrix = _levenshtein_distance_matrix(string1, string2, is_damerau=is_damerau)
    i, j = dist_matrix.shape
    i -= 1
    j -= 1
    ops = list()
    while i != -1 and j != -1:
        if is_damerau:
            if i > 1 and j > 1 and string1[i - 1] == string2[j - 2] and string1[i - 2] == string2[j - 1]:
                if dist_matrix[i - 2, j - 2] < dist_matrix[i, j]:
                    ops.insert(0, ('transposition', i - 1, i - 2))
                    i -= 2
                    j -= 2
                    continue
        index = np.argmin([dist_matrix[i - 1, j - 1], dist_matrix[i, j - 1], dist_matrix[i - 1, j]])
        if index == 0:
            if dist_matrix[i, j] > dist_matrix[i - 1, j - 1]:
                ops.insert(0, ('substitution', i - 1, j - 1))
            i -= 1
            j -= 1
        elif index == 1:
            ops.insert(0, ('insertion', i - 1, j - 1))
            j -= 1
        elif index == 2:
            ops.insert(0, ('deletion', i - 1, i - 1))
            i -= 1
    return ops

def execute_ops(ops, string1, string2):
    """
    executes edit operations on string 1
    Args:
        ops (list): edit operations
        string1 (str): word to edit
        string2 (str): correct word

    Returns:
        strings after operations (list)

    """
    strings = [string1]
    string = list(string1)
    shift = 0
    for op in ops:
        i, j = op[1], op[2]
        if op[0] == 'deletion':
            del string[i + shift]
            shift -= 1
        elif op[0] == 'insertion':
            string.insert(i + shift + 1, string2[j])
            shift += 1
        elif op[0] == 'substitution':
            string[i + shift] = string2[j]
        elif op[0] == 'transposition':
            string[i + shift], string[j + shift] = string[j + shift], string[i + shift]
        strings.append(''.join(string))
    return strings

lm2 = learn_language_model(["Rapunzel2.txt"], n=2, lm=None)
lm3 = learn_language_model(["Rapunzel2.txt"], n=3, lm=None)
dic = generateLexicon("Rapunzel2.txt")
err = create_error_distribution("errors.txt", dic)

n = 0
textdict = {}
for i in range(0, 9):
    textdict[i] = generate_text(lm2, 15, "the")
l = set(textdict.values())
n = len(l)

s = str(n) + ';'
try:
    s += (correct_word("whee", dic, err) + ';')  ##where
except Exception as e:
    s += 'error;'

try:
    s += (correct_word("child", dic, err) + ';')  ##child
except Exception as e:
    s += 'error;'

try:
    s += (correct_word("sha", dic, err) + ';')  ##she
except Exception as e:
    s += 'error;'

try:
    s += (correct_word("must", dic, err) + ';')  ##most
except Exception as e:
    s += 'error;'

try:
    s += (correct_word("why", dic, err) + ';')  ##who
except Exception as e:
    s += 'error;'

try:
    s += (correct_word("wave", dic, err) + ';')  ##weave
except Exception as e:
    s += 'error;'

try:
    s += (correct_word("bladder", dic, err) + ';')  ##ladder
except Exception as e:
    s += 'error;'

try:
    s += (correct_word("godl", dic, err) + ';')  ##gold
except Exception as e:
    s += 'error;'

try:
    s += (correct_word("as", dic, err) + ';')  ##as
except Exception as e:
    s += 'error;'

try:
    s += (correct_word("evenings", dic, err) + ';')  ##evening
except Exception as e:
    s += 'error;'

try:
    s += (correct_sentence("evey evening that beautiful bird cane by", lm2, err, 2,
                           0.75) + ';')  # every evening that beautiful bird came by
except Exception as e:
    s += 'error;'

try:
    s += (
        correct_sentence("evey evening that beautiful bird cane by", lm3, err, 2,
                         0.5) + ';')  # every evening that beautiful bird came by
except Exception as e:
    s += 'error;'

try:
    s += (correct_sentence("came ni with joy", lm2, err, 2, 0.7) + ';')  # came in with joy
except Exception as e:
    s += 'error;'

try:
    s += (correct_sentence("came ni with joy", lm3, err, 2, 0.7) + ';')  # came in with joy
except Exception as e:
    s += 'error;'

try:
    s += (correct_sentence("threw her arm around his necks", lm2, err, 2, 0.6) + ';')  # threw her arms around his neck
except Exception as e:
    s += 'error;'

try:
    s += (correct_sentence("threw her arm around his necks", lm3, err, 2, 0.55) + ';')  # threw her arms around his neck
except Exception as e:
    s += 'error;'

try:
    s += (correct_sentence("singing with her sweat voice", lm2, err, 2, 0.6) + ';')  # singing with her sweet voice
except Exception as e:
    s += 'error;'

try:
    s += (correct_sentence("singing with her sweat voice", lm3, err, 2, 0.6) + ';')  # singing with her sweet voice
except Exception as e:
    s += 'error;'

try:
    s += (
    correct_sentence("the smae day he cried scornfully:", lm2, err, 2, 0.6) + ';')  # the same day she cried scornfully
except Exception as e:
    s += 'error;'

try:
    s += (correct_sentence("if a vary friendly manner", lm3, err, 2, 0.5) + ';')  # in a very firendly manner
except Exception as e:
    s += 'error;'

print s