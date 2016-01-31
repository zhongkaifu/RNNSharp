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
                    SimpleRNN sRNN = new SimpleRNN();

                    sRNN.setBPTT(m_modelSetting.Bptt + 1);
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
                if (m_modelSetting.ModelType == 0)
                {
                    SimpleRNN sForwardRNN = new SimpleRNN();
                    SimpleRNN sBackwardRNN = new SimpleRNN();

                    sForwardRNN.setBPTT(m_modelSetting.Bptt + 1);
                    sForwardRNN.setBPTTBlock(10);

                    sBackwardRNN.setBPTT(m_modelSetting.Bptt + 1);
                    sBackwardRNN.setBPTTBlock(10);

                    rnn = new BiRNN(sForwardRNN, sBackwardRNN);
                }
                else
                {
                    rnn = new BiRNN(new LSTMRNN(), new LSTMRNN());
                }
            }

            rnn.ModelDirection = (MODELDIRECTION)m_modelSetting.ModelDirection;
            rnn.ModelFile = m_modelSetting.ModelFile;
            rnn.SaveStep = m_modelSetting.SaveStep;
            rnn.MaxIter = m_modelSetting.MaxIteration;
            rnn.IsCRFTraining = m_modelSetting.IsCRFTraining;
            rnn.LearningRate = m_modelSetting.LearningRate;
            rnn.GradientCutoff = 15.0f;
            rnn.Dropout = m_modelSetting.Dropout;
            rnn.L1 = m_modelSetting.NumHidden;

            rnn.DenseFeatureSize = TrainingSet.DenseFeatureSize();
            rnn.L0 = TrainingSet.GetSparseDimension();
            rnn.L2 = TrainingSet.TagSize;

            rnn.initMem();
            
            //Create tag-bigram transition probability matrix only for sequence RNN mode
            if (m_modelSetting.IsCRFTraining)
            {
                rnn.setTagBigramTransition(TrainingSet.CRFLabelBigramTransition);
            }

            Logger.WriteLine(Logger.Level.info, "");

            Logger.WriteLine(Logger.Level.info, "[TRACE] Iterative training begins ...");
            double lastPPL = double.MaxValue;
            double lastAlpha = rnn.LearningRate;
            int iter = 0;
            while (true)
            {
                if (rnn.MaxIter > 0 && iter > rnn.MaxIter)
                {
                    Logger.WriteLine(Logger.Level.info, "We have trained this model {0} iteration, exit.");
                    break;
                }

                //Start to train model
                double ppl = rnn.TrainNet(TrainingSet, iter);

                //Validate the model by validated corpus
                bool betterValidateNet = false;
                if (rnn.ValidateNet(ValidationSet, iter) == true)
                {
                    //If current model is better than before, save it into file
                    Logger.WriteLine(Logger.Level.info, "Saving better model into file {0}...", m_modelSetting.ModelFile);
                    rnn.saveNetBin(m_modelSetting.ModelFile);

                    betterValidateNet = true;
                }

                if (ppl >= lastPPL && lastAlpha != rnn.LearningRate)
                {
                    //Although we reduce alpha value, we still cannot get better result.
                    Logger.WriteLine(Logger.Level.info, "Current perplexity({0}) is larger than the previous one({1}). End training early.", ppl, lastPPL);
                    Logger.WriteLine(Logger.Level.info, "Current alpha: {0}, the previous alpha: {1}", rnn.LearningRate, lastAlpha);
                    break;
                }

                lastAlpha = rnn.LearningRate;
        //        if (betterValidateNet == false)
                if (ppl >= lastPPL)
                {
                    rnn.LearningRate = rnn.LearningRate / 2.0f;
                }

                lastPPL = ppl;

                iter++;
            }
        }
    }
}
