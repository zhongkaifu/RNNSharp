using AdvUtils;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>
namespace RNNSharp
{
    public class ModelSetting
    {
        public string ModelFile { get; set; }
        public int NumHidden { get; set; }
        public float LearningRate { get; set; }
        public float Dropout { get; set; }
        public int Bptt { get; set; }
        public int MaxIteration { get; set; }
        public bool IsCRFTraining { get; set; }
        public long SaveStep { get; set; }
        public int ModelType { get; set; }
        public int ModelDirection { get; set; }

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
            Logger.WriteLine("Hidden layer size： {0}", NumHidden);
            Logger.WriteLine("RNN-CRF: {0}", IsCRFTraining);
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
            NumHidden = 200;
            IsCRFTraining = true;
        }
    }
}
