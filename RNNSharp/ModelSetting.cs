using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AdvUtils;

namespace RNNSharp
{
    public class ModelSetting
    {
        public string GetModelFile() { return m_strModelFile; }
        public void SetModelFile(string modelFile) { m_strModelFile = modelFile; }

        public int GetNumHidden() { return m_NumHidden; }
        public void SetNumHidden(int n) { m_NumHidden = n; }

        public double GetLearningRate(){ return m_LearningRate; }
        public void SetLearningRate(double r) { m_LearningRate = r; }

        public double GetDropout() { return m_Dropout; }
        public void SetDropout(double r) { m_Dropout = r; }

        public int GetBptt() { return m_Bptt; }
        public void SetBptt(int n) { m_Bptt = n; }


        public int GetModelType() { return m_ModelType; }
        public void SetModelType(int n) { m_ModelType = n; }

        public int GetMaxIteration() { return m_MaxIteration; }
        public void SetMaxIteration(int i) { m_MaxIteration = i; }

        public virtual bool IsCRFTraining() { return m_bCRFTraining; }
        public void SetCRFTraining(bool s) { m_bCRFTraining = s; }

        public void SetDir(int dir)
        {
            m_iDir = dir;
        }

        public int GetModelDirection()
        {
            return m_iDir;
        }

        public void SetSaveStep(long savestep)
        {
            m_SaveStep = savestep;
        }

        public long GetSaveStep()
        {
            return m_SaveStep;
        }

        string m_strModelFile;
        int m_NumHidden;
        double m_LearningRate;
        double m_Dropout;
        int m_Bptt;
        int m_MaxIteration;
        bool m_bCRFTraining;
        long m_SaveStep;
        int m_ModelType;
        int m_iDir;

        public void DumpSetting()
        {
            Logger.WriteLine(Logger.Level.info, "Model File: {0}", m_strModelFile);
            if (m_ModelType == 0)
            {
                Logger.WriteLine(Logger.Level.info, "Model Structure: Simple RNN");
                Logger.WriteLine(Logger.Level.info, "BPTT: {0}", m_Bptt);
            }
            else if (m_ModelType == 1)
            {
                Logger.WriteLine(Logger.Level.info, "Model Structure: LSTM-RNN");
            }
            
            if (m_iDir == 0)
            {
                Logger.WriteLine(Logger.Level.info, "RNN Direction: Forward");
            }
            else
            {
                Logger.WriteLine(Logger.Level.info, "RNN Direction: Bi-directional");
            }

            Logger.WriteLine(Logger.Level.info, "Learning rate: {0}", m_LearningRate);
            Logger.WriteLine(Logger.Level.info, "Dropout: {0}", m_Dropout);
            Logger.WriteLine(Logger.Level.info, "Max Iteration: {0}", m_MaxIteration);
            Logger.WriteLine(Logger.Level.info, "Hidden layer size： {0}", m_NumHidden);
            Logger.WriteLine(Logger.Level.info, "RNN-CRF: {0}", m_bCRFTraining);
            if (m_SaveStep > 0)
            {
                Logger.WriteLine(Logger.Level.info, "Save temporary model after every {0} sentences", m_SaveStep);
            }
        }

        public ModelSetting()
        {
            m_MaxIteration = 20;
            m_Bptt = 4;
            m_LearningRate = 0.1;
            m_NumHidden = 200;
            m_bCRFTraining = true;
        }
    }
}
