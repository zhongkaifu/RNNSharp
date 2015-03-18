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
                    sRNN.setBPTTBlock(30);

                    rnn = sRNN;
                }
                else
                {
                    LSTMRNN lstmRNN = new LSTMRNN();
                    rnn = lstmRNN;
                }
            }
            else
            {
                BiRNN biRNN = new BiRNN(m_modelSetting.GetModelType());
                rnn = biRNN;
            }

            rnn.SetModelDirection(m_modelSetting.GetModelDirection());
            rnn.SetTrainingSet(m_TrainingSet);
            rnn.SetValidationSet(m_ValidationSet);
            rnn.SetModelFile(m_modelSetting.GetModelFile());
            rnn.SetSaveStep(m_modelSetting.GetSaveStep());
            rnn.SetMaxIter(m_modelSetting.GetMaxIteration());
            rnn.SetDynAlpha(m_modelSetting.GetIsDynAlpha());
            rnn.SetCRFTraining(m_modelSetting.IsCRFTraining());
            rnn.SetLearningRate(m_modelSetting.GetLearningRate());
            rnn.SetGradientCutoff(15.0);
            rnn.SetRegularization(m_modelSetting.GetRegularization());
            rnn.SetHiddenLayerSize(m_modelSetting.GetNumHidden());
            rnn.SetTagBigramTransitionWeight(m_modelSetting.GetTagTransitionWeight());

            rnn.initMem();
            
            //Create tag-bigram transition probability matrix only for sequence RNN mode
            if (m_modelSetting.IsCRFTraining() == true)
            {
                rnn.setTagBigramTransition(m_LabelBigramTransition);
            }

            Console.WriteLine();

            Console.WriteLine("[TRACE] Iterative training begins ...");
            while (rnn.ShouldTrainingStop() == false)
            {
                //Start to train model
                rnn.TrainNet();

                //Validate the model by validated corpus
                if (rnn.ValidateNet() == true)
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
                //}
            }
        }
    }
}
