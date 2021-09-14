import matlab.engine
import pandas as pd
import ibm_watson
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from ibm_watson import ToneAnalyzerV3
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

#-----New Tone Analysing.-----
def ToneNewValueRanger(ToneScore, ToneID):
    OldMax = 1
    OldMin = 0
    if ToneID == 'anger':
        NewMax = 100
        NewMin = 85
    elif ToneID == 'fear':
        NewMax = 90
        NewMin = 80
    elif ToneID == 'analytical':
        NewMax = 85
        NewMin = 70
    elif ToneID == 'negative':
        NewMax = 75
        NewMin = 60
    elif ToneID == 'tentive':
        NewMax = 70
        NewMin = 60
    elif ToneID == 'confident':
        NewMax = 70
        NewMin = 50
    elif ToneID == 'sadness':
        NewMax = 75
        NewMin = 40
    elif ToneID == 'joy':
        NewMax = 62
        NewMin = 25
    else:
        NewMax = 40
        NewMin = 0
    NewValue = (((ToneScore - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
    return round(NewValue, 2)

#-----Calling MATLAB-----
def MATLABSentimentAnalyzer(TheText):
    eng = matlab.engine.start_matlab()
    theDoc = eng.tokenizedDocument(TheText)
    compundScore = eng.vaderSentimentScores(theDoc)
    eng.quit()
    return compundScore

#-----Tokenzing words-----
def Tokens(text):
    all_stopwords = stopwords.words('english')
    sw_list = ['I',',']
    all_stopwords.extend(sw_list)
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in all_stopwords]
    return tokens_without_sw

#-----Counting Number of Bad Words-----
def NumBadWords(text_tokens, word_list):
    badWords = 0
    for i in text_tokens:
        theword = i.lower()
        for word in word_list:
            if theword == word:
                badWords += 1
    return badWords/len(text_tokens)

#-----Calling MATLAB Fuzzy Logic-----
def MATLABfuzzy(ObsceneWordsAnalyzer, ToneAnalyzer, SentimentAnalyzer, RadicalizedWordFreq):
    eng = matlab.engine.start_matlab()
    RadicalizedFuzzy = eng.readfis('FES9.fis')
    CrispInputs = matlab.double([ObsceneWordsAnalyzer, ToneAnalyzer, SentimentAnalyzer, RadicalizedWordFreq])
    CrispOutput = eng.evalfis(RadicalizedFuzzy, CrispInputs)
    return CrispOutput

#-----IBM sentimant Analysis-----
def IBMSentimantAnalysis(theText):
    response = natural_language_understanding.analyze(
        text=theText,
        features=Features(entities=EntitiesOptions(sentiment=True,limit=1))).get_result()
    return response

#-----Word Stemming function-----
def stemmer(Word):
    ps = PorterStemmer()
    w = Word.lower()
    return ps.stem(w)

#
def CheckingRadicalizedWords(TheSentence, Rd_word_list):
    tknztxt_1 = Tokens(TheSentence)
    LengthSentence = len(tknztxt_1)
    numRad = 0
    for I in tknztxt_1:
        stm = stemmer(I)
        for rd_word in Rd_word_list:
            if rd_word == stm:
                numRad += 1
    return numRad/LengthSentence

#-----The Fuzzy Classifier-----
def FuzzyClassifier(FuzzyOutput):
    if FuzzyOutput < 5:
        FuzzyRadicalied = 'NO'
    elif FuzzyOutput >= 5 and FuzzyOutput < 6:
        FuzzyRadicalied = 'MAYBE'
    elif FuzzyOutput > 6:
        FuzzyRadicalied = 'YES'
    return FuzzyRadicalied

#-----Tone Analyzer----
def ToneAnalyzer(TheText):
    ToneAz = ta.tone(TheText).get_result()
    return ToneAz

#-----The Final Function-----
def FinalFunction(OneDescription, TermList, sid, Rd_word_list):
    global line_num
    #Wastson Tone Analyzer
    Connection_Number = 0
    while True:
        Connection_Number += 1
        if Connection_Number > 3:
            print('!!!FATAL ERROR: Hmm Seems Like a IBM API or Network Issue!!!')
            break
        try:
        ##Calling the Tone Analyzer Func
            ToneResult = ToneAnalyzer(OneDescription)
            print("Tone Analyzing...\n")
            break
        except:
            print("OOPS: Connection Error Occurred")
            continue

    try:
        print('Getting New Tone Numbers...\n')
        ToneScore = ToneResult['document_tone']['tones'][0]['score']
        ToneID = ToneResult['document_tone']['tones'][0]['tone_id']
        New_Value = ToneNewValueRanger(ToneScore, ToneID)
    except:
        ToneScore = None
        ToneID = None
        New_Value = 0
        print('OOPS: Something Went Wrong while getting tone new numbers')

    #Swear Words frequency
    try:
        print("Checking Swearing Words...\n")
        WordTokens = Tokens(OneDescription)
        TheScore = NumBadWords(WordTokens, TermList)
    except:
        print('OOPS: Something went wrong with Word Tokenizing')
        TheScore = 0

    #Radicalized words Frequency
    try:
        print("Checking Radicalized Words...\n")
        RadFeqCal = CheckingRadicalizedWords(OneDescription, Rd_word_list)
    except:
        print('OOPS: Something Went Wrong while counting radicalized words')
        RadFeqCal = 0
    
    #Sentiment Analyzers
    try:
        print("Sentimant Analysing...\n")
        NLTKSentimentResult = sid.polarity_scores(OneDescription)
        SentimentCompund = NLTKSentimentResult['compound']
    except:
        print('OOPS: Something went wrong with Sentimant Analyzer')
        SentimentCompund = 0
    try:
        print('Using MATLAB...\n')
        FuzzyOutput = MATLABfuzzy(TheScore, New_Value, SentimentCompund, RadFeqCal)
    except:
        print('OOPS: something went wrong with Final Fuzzy output')
        FuzzyOutput = 5

    #Classifying  MATLAB output
    try:
        FuzzyActuallyRadicalied = FuzzyClassifier(FuzzyOutput)
    except:
        FuzzyActuallyRadicalied = 'MAYBE'

    #Showing Final Output
    try:
        print('\t',TheScore, New_Value, SentimentCompund, RadFeqCal)
        print('{}/332'.format(line_num))
        line_num += 1
        print(OneDescription)
        print('\tFuzzy output: {}'.format(FuzzyOutput))
        print("\n\n")
        return FuzzyActuallyRadicalied
    except:
        print("!!!!FATAL ERROR: something went terribly wrong!!!!")

#-----The Main Function-----
def main():
    #Reading the labled csv file
    try:
        df = pd.read_csv('Amalka1_GoodLabled.csv')
    except:
        print('OOPS: cannot find Labled csv file') 
    df1 = df[:2].copy()
    print('-->NOTE: Data Frame Size: {}\n\n'.format(df1.size))

    #Reading the Term Block File from FaceBook
    try:
        FBTerms = pd.read_csv('TermsToBlock.csv')
        TermList = FBTerms['words'] 
    except:
        print('OOPS: cannot find TermsToBlock.csv') 

    #Reading the Radicalized Word list csv file
    words = pd.read_csv('RadicalizedWords.csv')
    RDW = words['Words']
    Rd_word_list = []
    for r in RDW:
        ####Calling stemmer func
        this = stemmer(r)
        Rd_word_list.append(this)

    #IBM Watson Keys for Tone Analyzer and Natural Language Processing
    #Tone Analyzer
    apikey = '--'
    url = '--'
    #Natural Language Processing
    nlp_apikey = '--'
    nlp_url = '--'

    authenticator = IAMAuthenticator(apikey)
    global ta
    ta = ToneAnalyzerV3(version='2021-11-06', authenticator=authenticator)
    ta.set_service_url(url)


    authenticator = IAMAuthenticator(nlp_apikey)
    natural_language_understanding = NaturalLanguageUnderstandingV1(
        version='2021-08-01',
        authenticator=authenticator
    )
    natural_language_understanding.set_service_url(nlp_url)

    sid = SentimentIntensityAnalyzer()

    #Running the main code
    print('-->NOTE: please wait this may take some time to finish...\n')
    global line_num
    line_num = 0
    df1['FuzzySays'] = df1.loc[:,'Translated'].apply(lambda x:FinalFunction(x, TermList, sid, Rd_word_list))

main()
