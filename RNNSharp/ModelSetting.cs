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
            Logger.WriteLine(Logger.Level.info, "Model File: {0}", ModelFile);
            if (ModelType == 0)
            {
                Logger.WriteLine(Logger.Level.info, "Model Structure: Simple RNN");
                Logger.WriteLine(Logger.Level.info, "BPTT: {0}", Bptt);
            }
            else if (ModelType == 1)
            {
                Logger.WriteLine(Logger.Level.info, "Model Structure: LSTM-RNN");
            }
            
            if (ModelDirection == 0)
            {
                Logger.WriteLine(Logger.Level.info, "RNN Direction: Forward");
            }
            else
            {
                Logger.WriteLine(Logger.Level.info, "RNN Direction: Bi-directional");
            }

            Logger.WriteLine(Logger.Level.info, "Learning rate: {0}", LearningRate);
            Logger.WriteLine(Logger.Level.info, "Dropout: {0}", Dropout);
            Logger.WriteLine(Logger.Level.info, "Max Iteration: {0}", MaxIteration);
            Logger.WriteLine(Logger.Level.info, "Hidden layer size： {0}", NumHidden);
            Logger.WriteLine(Logger.Level.info, "RNN-CRF: {0}", IsCRFTraining);
            if (SaveStep > 0)
            {
                Logger.WriteLine(Logger.Level.info, "Save temporary model after every {0} sentences", SaveStep);
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
