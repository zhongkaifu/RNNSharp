using AdvUtils;
using System.Numerics;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>

namespace RNNSharp
{
    public class ModelSetting
    {
        public ModelSetting()
        {
            MaxIteration = 20;
            LearningRate = 0.1f;
            GradientCutoff = 15.0f;
            IncrementalTrain = false;
        }

        public int MiniBatchSize { get; set; }
        public bool IncrementalTrain { get; set; }
        public float LearningRate { get; set; }
        public int MaxIteration { get; set; }
        public bool IsConstAlpha { get; set; }
        public long SaveStep { get; set; }
        public int VQ { get; set; }
        public float GradientCutoff { get; set; }

        public void DumpSetting()
        {
            Logger.WriteLine($"Use const learning rate: {IsConstAlpha}");
            Logger.WriteLine($"Starting learning rate: {LearningRate}");
            Logger.WriteLine($"Max Iteration: {MaxIteration}");
            Logger.WriteLine($"SIMD: {Vector.IsHardwareAccelerated}, Size: {Vector<float>.Count * sizeof(float) * 8}bits");
            Logger.WriteLine($"Gradient cut-off: {GradientCutoff}");

            if (SaveStep > 0)
            {
                Logger.WriteLine($"Save temporary model after every {SaveStep} sentences");
            }

            Logger.WriteLine($"Incremental training: {IncrementalTrain}");
            Logger.WriteLine($"Mini batch size: {MiniBatchSize}");
        }
    }
}