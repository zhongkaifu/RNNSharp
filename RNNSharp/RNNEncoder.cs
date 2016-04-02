using AdvUtils;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>
namespace RNNSharp
{
    public class RNNEncoder
    {
        ModelSetting m_modelSetting;
        public DataSet TrainingSet { get; set; }
        public DataSet ValidationSet { get; set; }

        public RNNEncoder(ModelSetting modelSetting)
        {
            m_modelSetting = modelSetting;
        }

        public void Train()
        {
            RNN rnn;

            if (m_modelSetting.ModelDirection == 0)
            {
                if (m_modelSetting.ModelType == 0)
                {
                    SimpleRNN sRNN = new SimpleRNN(new SimpleLayer(m_modelSetting.NumHidden));

                    sRNN.setBPTT(m_modelSetting.Bptt + 1);
                    sRNN.setBPTTBlock(10);

                    rnn = sRNN;
                }
                else
                {
                    rnn = new LSTMRNN(new LSTMLayer(m_modelSetting.NumHidden));
                }
            }
            else
            {
                if (m_modelSetting.ModelType == 0)
                {
                    SimpleRNN sForwardRNN = new SimpleRNN(new SimpleLayer(m_modelSetting.NumHidden));
                    SimpleRNN sBackwardRNN = new SimpleRNN(new SimpleLayer(m_modelSetting.NumHidden));

                    sForwardRNN.setBPTT(m_modelSetting.Bptt + 1);
                    sForwardRNN.setBPTTBlock(10);

                    sBackwardRNN.setBPTT(m_modelSetting.Bptt + 1);
                    sBackwardRNN.setBPTTBlock(10);

                    rnn = new BiRNN(sForwardRNN, sBackwardRNN);
                }
                else
                {
                    rnn = new BiRNN(new LSTMRNN(new LSTMLayer(m_modelSetting.NumHidden)), new LSTMRNN(new LSTMLayer(m_modelSetting.NumHidden)));
                }
            }

            rnn.ModelDirection = (MODELDIRECTION)m_modelSetting.ModelDirection;
            rnn.bVQ = (m_modelSetting.VQ != 0) ? true : false;
            rnn.ModelFile = m_modelSetting.ModelFile;
            rnn.SaveStep = m_modelSetting.SaveStep;
            rnn.MaxIter = m_modelSetting.MaxIteration;
            rnn.IsCRFTraining = m_modelSetting.IsCRFTraining;
            rnn.LearningRate = m_modelSetting.LearningRate;
            rnn.GradientCutoff = m_modelSetting.GradientCutoff;
            rnn.Dropout = m_modelSetting.Dropout;
            rnn.L1 = m_modelSetting.NumHidden;

            rnn.DenseFeatureSize = TrainingSet.DenseFeatureSize();
            rnn.L0 = TrainingSet.GetSparseDimension();
            rnn.L2 = TrainingSet.TagSize;

            rnn.InitMem();
            
            //Create tag-bigram transition probability matrix only for sequence RNN mode
            if (m_modelSetting.IsCRFTraining)
            {
                rnn.setTagBigramTransition(TrainingSet.CRFLabelBigramTransition);
            }

            Logger.WriteLine("");

            Logger.WriteLine("Iterative training begins ...");
            double lastPPL = double.MaxValue;
            double lastAlpha = rnn.LearningRate;
            int iter = 0;
            while (true)
            {
                Logger.WriteLine("Cleaning training status...");
                rnn.CleanStatus();

                if (rnn.MaxIter > 0 && iter > rnn.MaxIter)
                {
                    Logger.WriteLine("We have trained this model {0} iteration, exit.");
                    break;
                }

                //Start to train model
                double ppl = rnn.TrainNet(TrainingSet, iter);
                if (ppl >= lastPPL && lastAlpha != rnn.LearningRate)
                {
                    //Although we reduce alpha value, we still cannot get better result.
                    Logger.WriteLine("Current perplexity({0}) is larger than the previous one({1}). End training early.", ppl, lastPPL);
                    Logger.WriteLine("Current alpha: {0}, the previous alpha: {1}", rnn.LearningRate, lastAlpha);
                    break;
                }
                lastAlpha = rnn.LearningRate;

                //Validate the model by validated corpus
                if (ValidationSet != null)
                {
                    Logger.WriteLine("Verify model on validated corpus.");
                    if (rnn.ValidateNet(ValidationSet, iter) == true)
                    {
                        //We got better result on validated corpus, save this model
                        Logger.WriteLine("Saving better model into file {0}...", m_modelSetting.ModelFile);
                        rnn.SaveModel(m_modelSetting.ModelFile);
                    }
                }
                else if (ppl < lastPPL)
                {
                    //We don't have validate corpus, but we get a better result on training corpus
                    //We got better result on validated corpus, save this model
                    Logger.WriteLine("Saving better model into file {0}...", m_modelSetting.ModelFile);
                    rnn.SaveModel(m_modelSetting.ModelFile);
                }
                
                if (ppl >= lastPPL)
                {
                    //We cannot get a better result on training corpus, so reduce learning rate
                    rnn.LearningRate = rnn.LearningRate / 2.0f;
                }

                lastPPL = ppl;

                iter++;
            }
        }
    }
}
