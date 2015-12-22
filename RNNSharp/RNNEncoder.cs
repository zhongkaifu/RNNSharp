using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace RNNSharp
{
    public class RNNEncoder
    {
        ModelSetting m_modelSetting;
        DataSet m_TrainingSet;
        DataSet m_ValidationSet;
        List<List<double>> m_LabelBigramTransition;

        public void SetLabelBigramTransition(List<List<double>> m)
        {
            m_LabelBigramTransition = m;
        }

        public RNNEncoder(ModelSetting modelSetting)
        {
            m_modelSetting = modelSetting;
        }


        public void SetTrainingSet(DataSet train)
        {
            m_TrainingSet = train;
        }
        public void SetValidationSet(DataSet validation)
        {
            m_ValidationSet = validation;
        }

        public void Train()
        {
            RNN rnn;

            if (m_modelSetting.GetModelDirection() == 0)
            {
                if (m_modelSetting.GetModelType() == 0)
                {
                    SimpleRNN sRNN = new SimpleRNN();

                    sRNN.setBPTT(m_modelSetting.GetBptt() + 1);
                    sRNN.setBPTTBlock(10);

                    rnn = sRNN;
                }
                else
                {
                    rnn = new LSTMRNN();
                }
            }
            else
            {
                if (m_modelSetting.GetModelType() == 0)
                {
                    SimpleRNN sForwardRNN = new SimpleRNN();
                    SimpleRNN sBackwardRNN = new SimpleRNN();

                    sForwardRNN.setBPTT(m_modelSetting.GetBptt() + 1);
                    sForwardRNN.setBPTTBlock(10);

                    sBackwardRNN.setBPTT(m_modelSetting.GetBptt() + 1);
                    sBackwardRNN.setBPTTBlock(10);

                    rnn = new BiRNN(sForwardRNN, sBackwardRNN);
                }
                else
                {
                    rnn = new BiRNN(new LSTMRNN(), new LSTMRNN());
                }
            }

            //Set model type
            rnn.SetModelDirection(m_modelSetting.GetModelDirection());

            //Set feature dimension
            rnn.SetFeatureDimension(m_TrainingSet.GetDenseDimension(), 
                m_TrainingSet.GetSparseDimension(), 
                m_TrainingSet.GetTagSize());


            rnn.SetModelFile(m_modelSetting.GetModelFile());
            rnn.SetSaveStep(m_modelSetting.GetSaveStep());
            rnn.SetMaxIter(m_modelSetting.GetMaxIteration());
            rnn.SetCRFTraining(m_modelSetting.IsCRFTraining());
            rnn.SetLearningRate(m_modelSetting.GetLearningRate());
            rnn.SetGradientCutoff(15.0);
            rnn.SetDropout(m_modelSetting.GetDropout());
            rnn.SetHiddenLayerSize(m_modelSetting.GetNumHidden());

            rnn.initMem();
            
            //Create tag-bigram transition probability matrix only for sequence RNN mode
            if (m_modelSetting.IsCRFTraining() == true)
            {
                rnn.setTagBigramTransition(m_LabelBigramTransition);
            }

            Console.WriteLine();

            Console.WriteLine("[TRACE] Iterative training begins ...");
            double lastPPL = double.MaxValue;
            double lastAlpha = rnn.Alpha;
            int iter = 0;
            while (true)
            {
                if (rnn.MaxIter > 0 && iter > rnn.MaxIter)
                {
                    Console.WriteLine("We have trained this model {0} iteration, exit.");
                    break;
                }

                //Start to train model
                double ppl = rnn.TrainNet(m_TrainingSet, iter);

                //Validate the model by validated corpus
                if (rnn.ValidateNet(m_ValidationSet) == true)
                {
                    //If current model is better than before, save it into file
                    Console.Write("Saving better model into file {0}...", m_modelSetting.GetModelFile());
                    rnn.saveNetBin(m_modelSetting.GetModelFile());
                    Console.WriteLine("Done.");
                }
                //else
                //{
                //    Console.Write("Loading previous best model from file {0}...", m_modelSetting.GetModelFile());
                //    rnn.loadNetBin(m_modelSetting.GetModelFile());
                //    Console.WriteLine("Done.");

                //    lastAlpha = rnn.Alpha;
                //    rnn.Alpha = rnn.Alpha / 2.0;
                //}


                if (ppl >= lastPPL && lastAlpha != rnn.Alpha)
                {
                    //Although we reduce alpha value, we still cannot get better result.
                    Console.WriteLine("Current perplexity({0}) is larger than the previous one({1}). End training early.", ppl, lastPPL);
                    Console.WriteLine("Current alpha: {0}, the previous alpha: {1}", rnn.Alpha, lastAlpha);
                    break;
                }

                lastAlpha = rnn.Alpha;
                if (ppl >= lastPPL)
                {
                    rnn.Alpha = rnn.Alpha / 2.0;
                }

                lastPPL = ppl;

                iter++;
            }
        }
    }
}
