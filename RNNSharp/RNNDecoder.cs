using AdvUtils;
using RNNSharp.Networks;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>

namespace RNNSharp
{
    public class RNNDecoder
    {
       private ConcurrentQueue<RNN<Sequence>> qRNNs;
        public Config Config;

        public RNNDecoder(Config config)
        {
            Config = config;
            RNN<Sequence> rnn = RNN<Sequence>.CreateRNN(Config.NetworkType);
            rnn.LoadModel(config.ModelFilePath);
            rnn.MaxSeqLength = config.MaxSequenceLength;

            Logger.WriteLine("CRF Model: {0}", rnn.IsCRFTraining);
            Logger.WriteLine($"Max Sequence Length: {rnn.MaxSeqLength}");
            Logger.WriteLine($"Processor Count: {Environment.ProcessorCount}");

            qRNNs = new ConcurrentQueue<RNN<Sequence>>();
            for (var i = 0; i < Environment.ProcessorCount; i++)
            {
                qRNNs.Enqueue(rnn.Clone());
            }
        }

        private RNN<Sequence> GetRNNInstance()
        {
            RNN<Sequence> r = null;
            while (qRNNs.TryDequeue(out r) == false)
            {
                Thread.Yield();
            }

            return r;
        }

        private void FreeRNNInstance(RNN<Sequence> r)
        {
            qRNNs.Enqueue(r);
        }

        public NETWORKTYPE NetworkType => Config.NetworkType;

        public int[][] ProcessNBest(Sentence sent, int nbest)
        {
            var rnn = GetRNNInstance();
            if (rnn.IsCRFTraining == false)
            {
                throw new ArgumentException("N-best result is only for RNN-CRF model.");
            }

            var predicted = rnn.DecodeNBestCRF(sent, Config, nbest);

            FreeRNNInstance(rnn);

            return predicted;
        }

        public int[] Process(Sentence sent)
        {
            int[] predicted = null;
            var rnn = GetRNNInstance();

            if (sent.TokensList.Count >= rnn.MaxSeqLength)
            {
                Logger.WriteLine($"The length of given sentnce is larger than {rnn.MaxSeqLength}, so ignore it: {sent.ToString()}");
            }
            else
            {
                predicted = rnn.IsCRFTraining ? rnn.DecodeCRF(sent, Config) : rnn.DecodeNN(sent, Config);
            }

            FreeRNNInstance(rnn);

            return predicted;
        }

        public float[][] ComputeTopHiddenLayerOutput(Sentence sent)
        {
            var rnn = GetRNNInstance();
            var seq = Config.BuildSequence(sent);
            float[][] output = rnn.ComputeTopHiddenLayerOutput(seq);
            FreeRNNInstance(rnn);

            return output;

        }


        public float[][] ComputeTopHiddenLayerOutput(Sequence seq)
        {

            var rnn = GetRNNInstance();
            float[][] output = rnn.ComputeTopHiddenLayerOutput(seq);
            FreeRNNInstance(rnn);
            return output;
        }

        public int GetTopHiddenLayerSize()
        {
            var rnn = GetRNNInstance();
            int n = rnn.GetTopHiddenLayerSize();
            FreeRNNInstance(rnn);
            return n;
        }
    }
}