using AdvUtils;
using System.Collections.Generic;
using System.Numerics;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>
namespace RNNSharp
{
    public enum LayerType
    {
        Softmax,
        NCESoftmax,
        BPTT,
        LSTM,
        DropOut
    }

    public class ModelSetting
    {
        public string TagFile { get; set; }
        public TagSet Tags { get; set; }
        public string ModelFile { get; set; }
        public List<int> HiddenLayerSizeList { get; set; }
        public float LearningRate { get; set; }
        public float Dropout { get; set; }
        public int Bptt { get; set; }
        public int MaxIteration { get; set; }
        public bool IsCRFTraining { get; set; }
        public bool IsSeq2SeqTraining { get; set; }
        public bool IsConstAlpha { get; set; }
        public long SaveStep { get; set; }
        public int ModelDirection { get; set; }
        public int VQ { get; set; }
        public float GradientCutoff { get; set; }
        public LayerType ModelType { get; set; }
        public LayerType OutputLayerType { get; set; }
        public int NCESampleSize { get; set; }

        public void DumpSetting()
        {
            Logger.WriteLine("Model File: {0}", ModelFile);
            Logger.WriteLine("Hidden Layer Type: {0}", ModelType.ToString());
            Logger.WriteLine("Output Layer Type: {0}", OutputLayerType.ToString());

            if (ModelDirection == 0)
            {
                Logger.WriteLine("RNN Direction: Forward");
            }
            else
            {
                Logger.WriteLine("RNN Direction: Bi-directional");
            }

            Logger.WriteLine("Seq2Seq model: {0}", IsSeq2SeqTraining);
            Logger.WriteLine("Use const learning rate: {0}", IsConstAlpha);
            Logger.WriteLine("Starting learning rate: {0}", LearningRate);
            Logger.WriteLine("Dropout: {0}", Dropout);
            Logger.WriteLine("Max Iteration: {0}", MaxIteration);
            Logger.WriteLine("Hidden layers: {0}", HiddenLayerSizeList.Count);
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
            HiddenLayerSizeList = null;
            IsCRFTraining = true;
        }
    }
}
