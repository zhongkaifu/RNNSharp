using AdvUtils;
using RNNSharp.Networks;
using System;
using System.Collections.Generic;
using System.Numerics;
using System.Threading;
using System.Threading.Tasks;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>

namespace RNNSharp
{
    public class RNNEncoder<T> where T : ISequence
    {
        private readonly Config featurizer;
        private readonly ModelSetting ModelSettings;

        public RNNEncoder(ModelSetting modelSetting, Config featurizer)
        {
            ModelSettings = modelSetting;
            this.featurizer = featurizer;
        }

        private NETWORKTYPE networkType => featurizer.NetworkType;

        private string modelFilePath => featurizer.ModelFilePath;

        private List<LayerConfig> hiddenLayersConfig => featurizer.HiddenLayersConfig;

        private LayerConfig outputLayerConfig => featurizer.OutputLayerConfig;

        public bool IsCRFTraining => featurizer.IsCRFTraining;
        private int maxSequenceLength => featurizer.MaxSequenceLength;

        public DataSet<T> TrainingSet { get; set; }
        public DataSet<T> ValidationSet { get; set; }

        int processedSequence = 0;
        int processedWordCnt = 0;
        int tknErrCnt = 0;
        int sentErrCnt = 0;

        public void Process(RNN<T> rnn, DataSet<T> trainingSet, RunningMode runningMode)
        {
            //Shffle training corpus
            trainingSet.Shuffle();

            for (var i = 0; i < trainingSet.SequenceList.Count; i++)
            {
                var pSequence = trainingSet.SequenceList[i];           

                int wordCnt = 0;
                if (pSequence is Sequence)
                {
                    wordCnt = (pSequence as Sequence).States.Length;
                }
                else
                {
                    SequencePair sp = pSequence as SequencePair;
                    if (sp.srcSentence.TokensList.Count > rnn.MaxSeqLength)
                    {
                        continue;
                    }

                    wordCnt = sp.tgtSequence.States.Length;
                    
                }

                if (wordCnt > rnn.MaxSeqLength)
                {
                    continue;
                }

                Interlocked.Add(ref processedWordCnt, wordCnt);

                int[] predicted;
                if (IsCRFTraining)
                {
                    predicted = rnn.ProcessSequenceCRF(pSequence as Sequence, runningMode);
                }
                else
                {
                    Matrix<float> m;
                    predicted = rnn.ProcessSequence(pSequence, runningMode, false, out m);
                }

                int newTknErrCnt;
                if (pSequence is Sequence)
                {
                    newTknErrCnt = GetErrorTokenNum(pSequence as Sequence, predicted);
                }
                else
                {
                    newTknErrCnt = GetErrorTokenNum((pSequence as SequencePair).tgtSequence, predicted);
                }

                Interlocked.Add(ref tknErrCnt, newTknErrCnt);
                if (newTknErrCnt > 0)
                {
                    Interlocked.Increment(ref sentErrCnt);
                }

                Interlocked.Increment(ref processedSequence);

                if (processedSequence % 1000 == 0)
                {
                    Logger.WriteLine("Progress = {0} ", processedSequence / 1000 + "K/" + TrainingSet.SequenceList.Count / 1000.0 + "K");
                    Logger.WriteLine(" Error token ratio = {0}%", (double)tknErrCnt / (double)processedWordCnt * 100.0);
                    Logger.WriteLine(" Error sentence ratio = {0}%", (double)sentErrCnt / (double)processedSequence * 100.0);
                }

                if (ModelSettings.SaveStep > 0 && processedSequence % ModelSettings.SaveStep == 0)
                {
                    //After processed every m_SaveStep sentences, save current model into a temporary file
                    Logger.WriteLine("Saving temporary model into file...");
                    rnn.SaveModel("model.tmp");
                }

                
            }
        }

        private int GetErrorTokenNum(Sequence seq, int[] predicted)
        {
            var tknErrCnt = 0;
            var numStates = seq.States.Length;
            for (var curState = 0; curState < numStates; curState++)
            {
                if (predicted[curState] != seq.States[curState].Label)
                {
                    tknErrCnt++;
                }
            }

            return tknErrCnt;
        }

        public void Train()
        {
            //Create neural net work
            Logger.WriteLine("Create a new network according settings in configuration file.");
            Logger.WriteLine($"Processor Count = {Environment.ProcessorCount}");

            RNN<T> rnn = RNN<T>.CreateRNN(networkType);
            if (ModelSettings.IncrementalTrain)
            {
                Logger.WriteLine($"Loading previous trained model from {modelFilePath}.");
                rnn.LoadModel(modelFilePath, true);
            }
            else
            {
                Logger.WriteLine("Create a new network.");
                rnn.CreateNetwork(hiddenLayersConfig, outputLayerConfig, TrainingSet, featurizer);
                //Create tag-bigram transition probability matrix only for sequence RNN mode
                if (IsCRFTraining)
                {
                    Logger.WriteLine("Initialize bigram transition for CRF output layer.");
                    rnn.setTagBigramTransition(TrainingSet.CRFLabelBigramTransition);
                }
            }

            rnn.MaxSeqLength = maxSequenceLength;
            int N = Environment.ProcessorCount;
            List<DataSet<T>> dataSets = TrainingSet.Split(N);
            List<RNN<T>> rnns = new List<RNN<T>>();
            rnns.Add(rnn);

            for (int i = 1; i < N; i++)
            {
                rnns.Add(rnn.Clone());
            }

            for (int i = 0; i < N; i++)
            {
                if (IsCRFTraining)
                {
                    dataSets[i].BuildLabelBigramTransition();
                }

                //Assign model settings to RNN
                rnns[i].bVQ = ModelSettings.VQ != 0 ? true : false;
                rnns[i].IsCRFTraining = IsCRFTraining;
            }

            //Initialize RNNHelper
            RNNHelper.LearningRate = ModelSettings.LearningRate;
            RNNHelper.vecNormalLearningRate = new Vector<float>(RNNHelper.LearningRate);

            RNNHelper.GradientCutoff = ModelSettings.GradientCutoff;
            RNNHelper.vecMaxGrad = new Vector<float>(RNNHelper.GradientCutoff);
            RNNHelper.vecMinGrad = new Vector<float>(-RNNHelper.GradientCutoff);

            RNNHelper.IsConstAlpha = ModelSettings.IsConstAlpha;

            Logger.WriteLine("");

            Logger.WriteLine("Iterative training begins ...");
            var bestTrainTknErrCnt = long.MaxValue;
            var bestValidTknErrCnt = long.MaxValue;
            var lastAlpha = RNNHelper.LearningRate;
            var iter = 0;
            ParallelOptions parallelOption = new ParallelOptions();
            parallelOption.MaxDegreeOfParallelism = -1;
            while (true)
            {
                processedSequence = 0;
                processedWordCnt = 0;
                tknErrCnt = 0;
                sentErrCnt = 0;

                if (ModelSettings.MaxIteration > 0 && iter > ModelSettings.MaxIteration)
                {
                    Logger.WriteLine("We have trained this model {0} iteration, exit.");
                    break;

                }

                Logger.WriteLine($"Start to training {iter} iteration. learning rate = {RNNHelper.LearningRate}");
                Parallel.For(0, N, i =>
                {
                    rnns[i].CleanStatus();
                    Process(rnns[i], dataSets[i], RunningMode.Training);
                });

                Logger.WriteLine($"End {iter} iteration.");
                Logger.WriteLine("");

                if (tknErrCnt >= bestTrainTknErrCnt && lastAlpha != RNNHelper.LearningRate)
                {
                    //Although we reduce alpha value, we still cannot get better result.
                    Logger.WriteLine(
                        $"Current token error count({(double)tknErrCnt / (double)processedWordCnt * 100.0}%) is larger than the previous one({(double)bestTrainTknErrCnt / (double)processedWordCnt * 100.0}%) on training set. End training early.");
                    Logger.WriteLine("Current alpha: {0}, the previous alpha: {1}", RNNHelper.LearningRate, lastAlpha);
                    break;
                }
                lastAlpha = RNNHelper.LearningRate;

                int trainTknErrCnt = tknErrCnt;
                //Validate the model by validated corpus
                if (ValidationSet != null)
                {
                    processedSequence = 0;
                    processedWordCnt = 0;
                    tknErrCnt = 0;
                    sentErrCnt = 0;

                    Logger.WriteLine("Verify model on validated corpus.");
                    Process(rnn, ValidationSet, RunningMode.Validate);
                    Logger.WriteLine("End model verification.");
                    Logger.WriteLine("");

                    if (tknErrCnt < bestValidTknErrCnt)
                    {
                        //We got better result on validated corpus, save this model
                        Logger.WriteLine($"Saving better model into file {modelFilePath}, since we got a better result on validation set.");
                        Logger.WriteLine($"Error token percent: {(double)tknErrCnt / (double)processedWordCnt * 100.0}%, Error sequence percent: {(double)sentErrCnt / (double)processedSequence * 100.0}%");
                        Logger.WriteLine("");

                        rnn.SaveModel(modelFilePath);
                        bestValidTknErrCnt = tknErrCnt;
                    }
                }
                else if (trainTknErrCnt < bestTrainTknErrCnt)
                {
                    //We don't have validate corpus, but we get a better result on training corpus
                    //We got better result on validated corpus, save this model
                    Logger.WriteLine($"Saving better model into file {modelFilePath}, although validation set doesn't exist, we have better result on training set.");
                    Logger.WriteLine($"Error token percent: {(double)trainTknErrCnt / (double)processedWordCnt * 100.0}%, Error sequence percent: {(double)sentErrCnt / (double)processedSequence * 100.0}%");
                    Logger.WriteLine("");

                    rnn.SaveModel(modelFilePath);
                }
                
                if (trainTknErrCnt >= bestTrainTknErrCnt)
                {
                    //We don't have better result on training set, so reduce learning rate
                    RNNHelper.LearningRate = RNNHelper.LearningRate / 2.0f;
                }
                else
                {
                    bestTrainTknErrCnt = trainTknErrCnt;
                }

                iter++;
            }
        }
    }
}