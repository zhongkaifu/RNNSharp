using AdvUtils;
using RNNSharp.Networks;
using System;
using System.Collections.Concurrent;
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

        int updatingWeights = 0;
        int processMiniBatch = 0;

        ParallelOptions parallelOptions;

        /// <summary>
        /// Process entire corpus set by given RNN
        /// </summary>
        /// <param name="rnns"></param>
        /// <param name="corpusSet"></param>
        /// <param name="runningMode"></param>
        public void Process(List<RNN<T>> rnns, DataSet<T> corpusSet, RunningMode runningMode)
        {
            parallelOptions = new ParallelOptions();
            parallelOptions.MaxDegreeOfParallelism = Environment.ProcessorCount;
            processedSequence = 0;
            processedWordCnt = 0;
            tknErrCnt = 0;
            sentErrCnt = 0;

            corpusSet.Shuffle();

            //Add RNN instance into job queue
            ConcurrentQueue<RNN<T>> qRNNs = new ConcurrentQueue<RNN<T>>();
            foreach (var rnn in rnns)
            {
                qRNNs.Enqueue(rnn);
            }

            Parallel.For(0, corpusSet.SequenceList.Count, parallelOptions, i =>
            {
                //Get a free RNN instance for running
                RNN<T> rnn;
                if (qRNNs.TryDequeue(out rnn) == false)
                {
                    //The queue is empty, so we clone a new one
                    rnn = rnns[0].Clone();
                    Logger.WriteLine("Cloned a new RNN instance for training.");
                }

                var pSequence = corpusSet.SequenceList[i];

                //Calcuate how many tokens we are going to process in this sequence
                int tokenCnt = 0;
                if (pSequence is Sequence)
                {
                    tokenCnt = (pSequence as Sequence).States.Length;
                }
                else
                {
                    SequencePair sp = pSequence as SequencePair;
                    if (sp.srcSentence.TokensList.Count > rnn.MaxSeqLength)
                    {
                        qRNNs.Enqueue(rnn);
                        return;
                    }
                    tokenCnt = sp.tgtSequence.States.Length;
                }

                //This sequence is too long, so we ignore it
                if (tokenCnt > rnn.MaxSeqLength)
                {
                    qRNNs.Enqueue(rnn);
                    return;
                }

                //Run neural network
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

                //Update counters
                Interlocked.Add(ref processedWordCnt, tokenCnt);
                Interlocked.Increment(ref processedSequence);
                Interlocked.Increment(ref processMiniBatch);

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

                //Update weights
                //We only allow one thread to update weights, and other threads keep running to train or predict given sequences
                //Note: we don't add any lock when updating weights and deltas for weights in order to improve performance singificantly, 
                //so that means race condition will happen and it's okay for us.
                if (runningMode == RunningMode.Training && processMiniBatch > 0 && processMiniBatch % ModelSettings.MiniBatchSize == 0 && updatingWeights == 0)
                {
                    Interlocked.Increment(ref updatingWeights);
                    if (updatingWeights == 1)
                    {
                        rnn.UpdateWeights();
                        Interlocked.Exchange(ref processMiniBatch, 0);
                    }
                    Interlocked.Decrement(ref updatingWeights);
                }

                //Show progress information
                if (processedSequence % 1000 == 0)
                {
                    Logger.WriteLine("Progress = {0} ", processedSequence / 1000 + "K/" + corpusSet.SequenceList.Count / 1000.0 + "K");
                    Logger.WriteLine(" Error token ratio = {0}%", (double)tknErrCnt / (double)processedWordCnt * 100.0);
                    Logger.WriteLine(" Error sentence ratio = {0}%", (double)sentErrCnt / (double)processedSequence * 100.0);
                }

                //Save intermediate model file
                if (ModelSettings.SaveStep > 0 && processedSequence % ModelSettings.SaveStep == 0)
                {
                    //After processed every m_SaveStep sentences, save current model into a temporary file
                    Logger.WriteLine("Saving temporary model into file...");
                    try
                    {
                        rnn.SaveModel("model.tmp");
                    }
                    catch (Exception err)
                    {
                        Logger.WriteLine($"Fail to save temporary model into file. Error: {err.Message.ToString()}");
                    }
                }

                qRNNs.Enqueue(rnn);
            });

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
                    rnn.InitializeCRFWeights(TrainingSet.CRFLabelBigramTransition);
                }
            }

            rnn.MaxSeqLength = maxSequenceLength;
            rnn.bVQ = ModelSettings.VQ != 0 ? true : false;
            rnn.IsCRFTraining = IsCRFTraining;
            if (rnn.IsCRFTraining)
            {
                rnn.InitializeCRFVariablesForTraining();
            }

            int N = Environment.ProcessorCount * 2;
            List<RNN<T>> rnns = new List<RNN<T>>();
            rnns.Add(rnn);

            for (int i = 1; i < N; i++)
            {
                rnns.Add(rnn.Clone());
            }

            //Initialize RNNHelper
            RNNHelper.LearningRate = ModelSettings.LearningRate;
            RNNHelper.vecNormalLearningRate = new Vector<float>(RNNHelper.LearningRate);

            RNNHelper.GradientCutoff = ModelSettings.GradientCutoff;
            RNNHelper.vecMaxGrad = new Vector<float>(RNNHelper.GradientCutoff);
            RNNHelper.vecMinGrad = new Vector<float>(-RNNHelper.GradientCutoff);
            RNNHelper.IsConstAlpha = ModelSettings.IsConstAlpha;
            RNNHelper.MiniBatchSize = ModelSettings.MiniBatchSize;

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
                if (ModelSettings.MaxIteration > 0 && iter > ModelSettings.MaxIteration)
                {
                    Logger.WriteLine("We have trained this model {0} iteration, exit.");
                    break;
                }

                var start = DateTime.Now;
                Logger.WriteLine($"Start to training {iter} iteration. learning rate = {RNNHelper.LearningRate}");

                //Clean all RNN instances' status for training
                foreach (var r in rnns)
                {
                    r.CleanStatusForTraining();
                }
                Process(rnns, TrainingSet, RunningMode.Training);

                var duration = DateTime.Now.Subtract(start);

                Logger.WriteLine($"End {iter} iteration. Time duration = {duration}");
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
                    Logger.WriteLine("Verify model on validated corpus.");
                    Process(rnns, ValidationSet, RunningMode.Validate);
                    Logger.WriteLine("End model verification.");
                    Logger.WriteLine("");

                    if (tknErrCnt < bestValidTknErrCnt)
                    {
                        //We got better result on validated corpus, save this model
                        Logger.WriteLine($"Saving better model into file {modelFilePath}, since we got a better result on validation set.");
                        Logger.WriteLine($"Error token percent: {(double)tknErrCnt / (double)processedWordCnt * 100.0}%, Error sequence percent: {(double)sentErrCnt / (double)processedSequence * 100.0}%");

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

                    rnn.SaveModel(modelFilePath);
                }

                Logger.WriteLine("");

                if (trainTknErrCnt >= bestTrainTknErrCnt)
                {
                    //We don't have better result on training set, so reduce learning rate
                    RNNHelper.LearningRate = RNNHelper.LearningRate / 2.0f;
                    RNNHelper.vecNormalLearningRate = new Vector<float>(RNNHelper.LearningRate);
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