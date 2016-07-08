using AdvUtils;
using System.Collections.Generic;
using System.Numerics;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>
namespace RNNSharp
{
    public class ModelSetting
    {
        public string TagFile { get; set; }
        public TagSet Tags { get; set; }
        public DataSet TrainDataSet { get; set; }
        public string ModelFile { get; set; }
        public List<int> NumHidden { get; set; }
        public float LearningRate { get; set; }
        public float Dropout { get; set; }
        public int Bptt { get; set; }
        public int MaxIteration { get; set; }
        public bool IsCRFTraining { get; set; }
        public long SaveStep { get; set; }
        public int ModelType { get; set; }
        public int ModelDirection { get; set; }
        public int VQ { get; set; }
        public float GradientCutoff { get; set; }

        public void DumpSetting()
        {
            Logger.WriteLine("Model File: {0}", ModelFile);
            if (ModelType == 0)
            {
                Logger.WriteLine("Model Structure: Simple RNN");
                Logger.WriteLine("BPTT: {0}", Bptt);
            }
            else if (ModelType == 1)
            {
                Logger.WriteLine("Model Structure: LSTM-RNN");
            }
            
            if (ModelDirection == 0)
            {
                Logger.WriteLine("RNN Direction: Forward");
            }
            else
            {
                Logger.WriteLine("RNN Direction: Bi-directional");
            }

            Logger.WriteLine("Learning rate: {0}", LearningRate);
            Logger.WriteLine("Dropout: {0}", Dropout);
            Logger.WriteLine("Max Iteration: {0}", MaxIteration);
            Logger.WriteLine("Hidden layers: {0}", NumHidden.Count);
            Logger.WriteLine("RNN-CRF: {0}", IsCRFTraining);
            Logger.WriteLine("SIMD: {0}, Size: {1}bits", System.Numerics.Vector.IsHardwareAccelerated, 
                Vector<double>.Count * sizeof(double) * 8);
            Logger.WriteLine("Gradient cut-off: {0}", GradientCutoff);
            if (SaveStep > 0)
            {
                Logger.WriteLine("Save temporary model after every {0} sentences", SaveStep);
            }
        }

        public ModelSetting()
        {
            MaxIteration = 20;
            Bptt = 4;
            LearningRate = 0.1f;
            GradientCutoff = 15.0f;
            NumHidden = null;
            IsCRFTraining = true;
        }
    }
}
