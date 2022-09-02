package analysis

import (
	"math"
	"strings"

	"github.com/andreas-holm/codecamp22/cmd/codecamp22/experiment"
	"github.com/andreas-holm/codecamp22/cmd/codecamp22/parse"
)

var useTextMessageAsTest bool

type Preprocessor interface {
	Process(ex *experiment.Experiment)
}

type Analysis struct {
	Name        string
	TrainingSet TrainingSet
	TestSet     TestSet
	FoundClass  experiment.Class
}

func (a Analysis) TestTestData(set experiment.TestSet) TestSet {
	//Create struct Testset
	results := TestSet{
		MessageTotal:          len(set.Cases),
		CorrectHam:            0,
		CorrectSpam:           0,
		IncorrectHam:          0,
		IncorrectSpam:         0,
		PercentageCorrectHam:  0,
		PercentageCorrectSpam: 0,
	}

	var hamScore float64
	var spamScore float64
	//loo over all sentences in the test data set
	for _, sms := range set.Cases {
		//Score to
		hamScore, spamScore = 0, 0
		//Split the message into seperate words
		words := strings.Split(sms.Text, " ")
		//Loop over all the words in a message
		for _, word := range words {
			// skip word if it isn't in the vocabulary
			if !a.TrainingSet.Vocabulary.Contains(word) {
				continue
			}

			// ham (add logs to prevent underflow of float with lots of multiplying)
			//Log of a*b = log a + log b so this will be added not multiplicated
			//Take wordprobability matrix for ham and check the probability for this word being in the class ham
			hamScore = hamScore + math.Log10(a.TrainingSet.Ham.WordProbabilities[word])

			// spam (add logs to prevent underflow of float with lots of multiplying)
			//Take wordprobability matrix for Spam and check the probability for this word being in the class spam
			spamScore = spamScore + math.Log10(a.TrainingSet.Spam.WordProbabilities[word])
		}
		// When we got all the summed upp scores for all wordprobabilities we unlog
		hamScore = math.Pow(10, hamScore)
		spamScore = math.Pow(10, spamScore)

		// if the algorithm says this is *ham*
		if hamScore > spamScore {
			if sms.Class == experiment.HamClass {
				results.CorrectHam = results.CorrectHam + 1
			} else {
				results.IncorrectHam = results.IncorrectHam + 1
			}
		} else { // algorithm says this is *spam*
			if sms.Class == experiment.SpamClass {
				results.CorrectSpam = results.CorrectSpam + 1
			} else {
				results.IncorrectSpam = results.IncorrectSpam + 1
			}
		}
	}
	//Check percentages correct of each class which is correctClassifiedHam/(correctClassifiedHam + incorrectClassifiedSpam)
	//For the percentage of correct ham
	results.PercentageCorrectHam = float64(results.CorrectHam) / float64(results.CorrectHam+results.IncorrectSpam)
	results.PercentageCorrectSpam = float64(results.CorrectSpam) / float64(results.CorrectSpam+results.IncorrectHam)

	return results
}

func (a Analysis) TestTextMessage(ex experiment.Experiment) experiment.Class {

	var hamScore float64
	var spamScore float64
	textMessage := ex.TextMessage
	//Score to get calculated and compared which class it belongs to
	hamScore, spamScore = 0, 0
	//Split the message into seperate words
	words := strings.Split(textMessage, " ")
	//Loop over all the words in a message
	for _, word := range words {
		// skip word if it isn't in the vocabulary
		if !a.TrainingSet.Vocabulary.Contains(word) {
			continue
		}

		// ham (add logs to prevent underflow of float with lots of multiplying)
		//Log of a*b = log a + log b so this will be added not multiplicated
		//Take wordprobability matrix for ham and check the probability for this word being in the class ham
		hamScore = hamScore + math.Log10(a.TrainingSet.Ham.WordProbabilities[word])

		// spam (add logs to prevent underflow of float with lots of multiplying)
		//Take wordprobability matrix for Spam and check the probability for this word being in the class spam
		spamScore = spamScore + math.Log10(a.TrainingSet.Spam.WordProbabilities[word])
	}
	// When we got all the summed upp scores for all wordprobabilities we unlog
	hamScore = math.Pow(10, hamScore)
	spamScore = math.Pow(10, spamScore)

	// if the algorithm says this is *ham*
	if hamScore > spamScore {
		return experiment.HamClass

	} else { // algorithm says this is *spam*
		return experiment.SpamClass
	}

}

type TestSet struct {
	MessageTotal          int
	CorrectHam            int
	CorrectSpam           int
	IncorrectHam          int
	IncorrectSpam         int
	PercentageCorrectHam  float64
	PercentageCorrectSpam float64
}

type TrainingSet struct {
	MessageTotal int
	Spam         Class
	Ham          Class
	Vocabulary   Vocabulary
}

type Vocabulary []string

func (v Vocabulary) Contains(word string) bool {
	for _, w := range v {
		if w == word {
			return true
		}
	}
	return false
}

type WordFrequency map[string]int

func (wf WordFrequency) Probability(v Vocabulary) Probability {
	//Vocabulary v is the list of all the existing words in the data
	//Word frequency wf is a map of the frequency of all the words in THIS class of messages.
	// I.E. in the SPAM class ["free" : 5]

	//Create a probability matrix/map p
	p := make(map[string]float64)

	//Loop over every vocabWord in vocabulary v
	for _, vocabWord := range v {
		//For every words frequency divide by the sum of the length of the words existing for THIS class
		//and the list of all the existing words in data file which will get the probability for this word
		//for THIS class we are checking.
		//Do this for every word and get probability map(matrix) for every word for this class.
		//+1 is a smoothing variable performed by the github owner who created the algorithm
		p[vocabWord] = float64(wf[vocabWord]+1) / float64(len(wf)+len(v))
	}

	return p
}

type Probability map[string]float64

type Class struct {
	// MessageTotal is the message total
	MessageTotal int
	// PofC is the probability of this class out of the total amount in the training set
	PofC float64
	// WordFrequency represents words and how many times they occur in this class
	WordFrequency WordFrequency
	// WordProbabilities is a lookup of words and their probability of occurring in this class
	WordProbabilities Probability
}

type Analyses []Analysis

func Run(ex experiment.Experiment, usetextMessageFlag bool) Analyses {
	var analyses Analyses
	useTextMessageAsTest = usetextMessageFlag
	// first analyze with no preprocessors
	defaultAnalysis := analysisFrom(ex)
	defaultAnalysis.Name = "Default Analysis (no preprocessing)"
	analyses = append(analyses, defaultAnalysis)

	// copy experiment for multiple types of preprocessing
	ex2, ex3, ex4, ex5 := ex, ex, ex, ex

	// analyze with punctuation removed
	parse.PreprocessRemovePunctuation{}.Process(&ex2)
	a2 := analysisFrom(ex2)
	a2.Name = "No Punctuation Analysis"
	analyses = append(analyses, a2)

	// analyze with stemmer
	parse.PreprocessStemmer{}.Process(&ex3)
	a3 := analysisFrom(ex3)
	a3.Name = "Stemmer Analysis"
	analyses = append(analyses, a3)

	// analyze with punctuation removed and then stemmer
	parse.PreprocessStemmer{}.Process(&ex4)
	parse.PreprocessRemovePunctuation{}.Process(&ex4)
	a4 := analysisFrom(ex4)
	a4.Name = "Stemmer and No Punctuation Analysis"
	analyses = append(analyses, a4)

	// analyze with common words removed
	parse.PreprocessRemoveCommonWords{}.Process(&ex5)
	a5 := analysisFrom(ex5)
	a5.Name = "Remove 100 Most Common English Words"
	analyses = append(analyses, a5)

	return analyses
}

func analysisFrom(ex experiment.Experiment) Analysis {
	//Total amount of training messages i.e. the sum of the length of the two classes in experiments
	totalTrainingMessages := len(ex.Classes.Ham) + len(ex.Classes.Spam)
	//Make a vocabulary, i.e. a list of all the words
	vocabulary := vocabularyFrom(ex.Classes.Ham, ex.Classes.Spam)
	//Calculate the word frequency map I.E. the frequency of every word in the messages of the class HAM.
	hamFrequency := wordFrequencyFrom(ex.Classes.Ham)
	//calculate the probability map(matrix) for every word to be in the HAM class.
	hamProbabilities := hamFrequency.Probability(vocabulary)
	//Calculate the word frequency map I.E. the frequency of every word in the messages of the class SPAM.
	spamFrequency := wordFrequencyFrom(ex.Classes.Spam)
	//calculate the probability map(matrix) for every word to be in the SPAM class.
	spamProbabilities := spamFrequency.Probability(vocabulary)
	//Create struct Analysis with all these parameters just calculated
	analysis := Analysis{
		TrainingSet: TrainingSet{
			MessageTotal: totalTrainingMessages,
			Ham: Class{
				MessageTotal:      len(ex.Classes.Ham),
				PofC:              float64(len(ex.Classes.Ham)) / float64(totalTrainingMessages),
				WordFrequency:     hamFrequency,
				WordProbabilities: hamProbabilities,
			},
			Spam: Class{
				MessageTotal:      len(ex.Classes.Spam),
				PofC:              float64(len(ex.Classes.Spam)) / float64(totalTrainingMessages),
				WordFrequency:     spamFrequency,
				WordProbabilities: spamProbabilities,
			},
			Vocabulary: vocabulary,
		},
	}
	//Create
	if useTextMessageAsTest {
		analysis.FoundClass = analysis.TestTextMessage(ex)
	} else {
		analysis.TestSet = analysis.TestTestData(ex.Test)
	}

	return analysis
}

func vocabularyFrom(messageLists ...[]string) Vocabulary {
	keys := make(map[string]bool)
	var vocabulary Vocabulary
	for _, messageList := range messageLists {
		for _, msg := range messageList {
			for _, word := range strings.Split(msg, " ") {
				if word == "" {
					continue
				}
				if _, exists := keys[word]; !exists {
					keys[word] = true
					vocabulary = append(vocabulary, word)
				}
			}
		}
	}

	return vocabulary
}

func wordFrequencyFrom(messageList []string) WordFrequency {
	//Make a map that holds the frequency of a word in this class
	frequency := make(map[string]int)
	//Loop all the messages in class
	for _, msg := range messageList {
		//Loop over all the words in a message
		for _, word := range strings.Split(msg, " ") {
			if word == "" {
				continue
			}

			occurrences, exists := frequency[word]
			//If the word exist in the map increase the frequency by one
			if exists {
				frequency[word] = occurrences + 1
			} else {
				//If not exist in map, initialize the word in the map and set 1 for this word
				frequency[word] = 1
			}
		}
	}

	return frequency
}
