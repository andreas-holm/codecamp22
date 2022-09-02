package main

import (
	"bufio"
	"flag"
	"fmt"
	"sort"

	"io"
	"math/rand"
	"os"
	"strings"
	"time"

	"github.com/andreas-holm/codecamp22/cmd/codecamp22/analysis"
	"github.com/andreas-holm/codecamp22/cmd/codecamp22/parse"
	"github.com/fatih/color"
)

// Whether to use a single textmessage and not testData
var useTextMessageAsTest bool = true
var numberOfCommonWordsForClass int = 5
var textMessageSpam string = "free 2 txt"
var textMessageHam string = "u have me and im in love with u 2"

func main() {

	run()
}

func run() {

	var flagFilename string
	flag.StringVar(&flagFilename, "file", "trainingData.data", "filename")
	var flagDelimiter string
	flag.StringVar(&flagDelimiter, "delimiter", "\t", "delimiter between class and words in data (default is tab)")
	flag.Parse()

	//Returns an experiment instance with seperated lines for SPAM and HAM respectivaley
	//Also returns a testSet of test cases. one test case is a label and messagetext
	exp, err := parse.FromFile(flagFilename, flagDelimiter, useTextMessageAsTest)
	//Set textMEssageData
	exp.TextMessage = textMessageHam

	if err != nil {
		fmt.Println("cannot parse file:", err)
		os.Exit(1)
	}

	analyses := analysis.Run(exp, useTextMessageAsTest)
	if useTextMessageAsTest {
		analyzeTextMessageClassification(analyses, exp.TextMessage)
	} else {
		analyzeTestDataClassification(analyses)
	}

	fmt.Println("\nDone.")
}

func analyzeTestDataClassification(analyses analysis.Analyses) {
	for _, a := range analyses {
		c := color.New(color.FgCyan).Add(color.Underline)
		c.Printf("Analysis: %s\n", a.Name)
		fmt.Println("Vocabulary has", len(a.TrainingSet.Vocabulary), "words")
		fmt.Println("\nTraining Set:")
		fmt.Printf("\t%d of %d messages were spam (%.2f%%)\n\n",
			a.TrainingSet.Spam.MessageTotal,
			a.TrainingSet.MessageTotal,
			a.TrainingSet.Spam.PofC*100)
		fmt.Println("Test Set:")
		fmt.Println("\tCorrect Ham:", a.TestSet.CorrectHam)
		fmt.Println("\tCorrect Spam:", a.TestSet.CorrectSpam)
		fmt.Println("\tIncorrect Ham (actually was spam):", a.TestSet.IncorrectHam)
		fmt.Println("\tIncorrect Spam (actually was ham):", a.TestSet.IncorrectSpam)
		fmt.Printf("\tPercentage Correct Ham: %.2f%%\n", a.TestSet.PercentageCorrectHam*100)
		fmt.Printf("\tPercentage Correct Spam: %.2f%%\n", a.TestSet.PercentageCorrectSpam*100)
		bold := color.New(color.FgGreen, color.Bold)
		bold.Printf("\tOverall Accuracy: %.2f%%\n",
			100*(float64(a.TestSet.CorrectSpam)+float64(a.TestSet.CorrectHam))/
				float64(a.TestSet.MessageTotal))
		fmt.Println()

	}
}

func analyzeTextMessageClassification(analyses analysis.Analyses, textMessage string) {
	for _, a := range analyses {
		c := color.New(color.FgCyan).Add(color.Underline)
		c.Printf("Analysis: %s\n", a.Name)
		fmt.Println("Vocabulary has", len(a.TrainingSet.Vocabulary), "words")
		fmt.Println("\nTraining Set:")
		fmt.Printf("\t%d of %d messages were spam (%.2f%%)\n\n",
			a.TrainingSet.Spam.MessageTotal,
			a.TrainingSet.MessageTotal,
			a.TrainingSet.Spam.PofC*100)

		bold := color.New(color.FgGreen, color.Bold)
		boldBlue := color.New(color.FgHiBlue, color.Bold)
		bold.Println("The five most common HAM words")
		mostCommonHamWords := getMostCommonWords(numberOfCommonWordsForClass, a.TrainingSet.Ham.WordFrequency)
		for i := len(mostCommonHamWords) - 1; i >= 0; i-- {
			boldBlue.Printf("Word")
			fmt.Println("\t\t", mostCommonHamWords[i].Word)
			boldBlue.Printf("Word frequancy")
			fmt.Println("\t", mostCommonHamWords[i].Frequency)
		}
		fmt.Println("")
		bold.Println("The five most common SPAM words")
		mostCommonSpamWords := getMostCommonWords(numberOfCommonWordsForClass, a.TrainingSet.Spam.WordFrequency)
		for i := len(mostCommonHamWords) - 1; i >= 0; i-- {
			boldBlue.Printf("Word")
			fmt.Println("\t\t", mostCommonSpamWords[i].Word)
			boldBlue.Printf("Word frequancy")
			fmt.Println("\t", mostCommonSpamWords[i].Frequency)
		}

		fmt.Println("")
		fmt.Println("")

		boldRed := color.New(color.FgRed, color.Bold)
		boldRed.Printf("Text Message: ")
		fmt.Println(textMessage)
		boldRed.Printf("Classifies as: ")
		fmt.Println(a.FoundClass.String())
		fmt.Println()

	}
}

func getMostCommonWords(amountOfWords int, wordFrequencies map[string]int) []WordFrequencyPair {

	sortedProbabilityList := make([]WordFrequencyPair, 0, amountOfWords)
	for key := range wordFrequencies {

		sortedProbabilityList = append(sortedProbabilityList,
			WordFrequencyPair{Word: key, Frequency: wordFrequencies[key]})
	}

	sort.SliceStable(sortedProbabilityList, func(i, j int) bool {
		return sortedProbabilityList[i].Frequency < sortedProbabilityList[j].Frequency
	})
	return sortedProbabilityList[len(sortedProbabilityList)-amountOfWords:]
}

type WordFrequencyPair struct {
	Word      string
	Frequency int
}

func parseTrainingData(filename, delimiter string) ([]pair, error) {
	//Try to open file
	file, err := os.Open(filename)
	//If error return
	if err != nil {
		return []pair{}, fmt.Errorf("reading %s: %w", filename, err)
	}
	defer file.Close()
	return parseData(file, delimiter)
}

func parseData(reader io.Reader, delimiter string) ([]pair, error) {
	//ratioToTrain := 0.75

	var labeledPairs []pair

	// append works on nil slices.

	lines := getEveryLineToList(reader)
	//read every line in file and add it to a list

	// randomize slice in-place so positioning of the line is non-dependant
	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(len(lines), func(i, j int) {
		//inner function to swap indexes
		lines[i], lines[j] = lines[j], lines[i]
	})
	//the number of lines to train with

	/*
		if !useTextMessageAsTest {
			numberToTrain := int(float64(len(lines)) * ratioToTrain)
		}
	*/

	for i, line := range lines {
		parts := strings.Split(line, delimiter)
		// eliminate lines without a class
		if len(parts) < 2 {
			fmt.Println("skipped line because it had not class:", parts[0])
			continue
		}
		thisClass, err := getClassType(parts[0])

		if err != nil {
			return labeledPairs, fmt.Errorf("checking class type on shuffled row %d: %w", i, err)
		}
		if len(parts[1]) == 0 {
			return labeledPairs, fmt.Errorf("empty text on shuffled row %d", i)
		}
		if thisClass == SpamClass {
			labeledPairs = append(labeledPairs, pair{class: SpamClass.toString(), lineString: parts[1]})
		}
		if thisClass == HamClass {
			labeledPairs = append(labeledPairs, pair{class: HamClass.toString(), lineString: parts[1]})
		}
		/*
			if !useTextMessageAsTest {
				if i < numberToTrain {
					// training sets
					if thisClass == SpamClass {
						ex.Classes.Spam = append(ex.Classes.Spam, parts[1])
					}
					if thisClass == HamClass {
						ex.Classes.Ham = append(ex.Classes.Ham, parts[1])
					}
				} else {
					// test sets
					testCase := experiment.TestCase{
						Class: thisClass,
						Text:  parts[1],
					}
					ex.Test.Cases = append(ex.Test.Cases, testCase)
				}
			}
		*/
	}

	return labeledPairs, nil
}

func getEveryLineToList(reader io.Reader) []string {
	scanner := bufio.NewScanner(reader)

	var lines []string
	for scanner.Scan() {
		line := scanner.Text()
		// skip empty lines
		if line == "" {
			continue
		}
		lines = append(lines, line)
	}
	return lines
}

func getClassType(str string) (Class, error) {
	switch str {
	case HamClass.toString():
		return HamClass, nil
	case SpamClass.toString():
		return SpamClass, nil
	default:
		return HamClass, fmt.Errorf("invalid class: %s", str)
	}
}

// Enum for the two classes
const (
	HamClass Class = iota
	SpamClass
)

type Class int

func (c Class) toString() string {
	switch c {
	case HamClass:
		return "ham"
	case SpamClass:
		return "spam"
	default:
		return ""
	}
}

type pair struct {
	class      string
	lineString string
}
