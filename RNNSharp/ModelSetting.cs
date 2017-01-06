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
        }

        public float LearningRate { get; set; }
        public int MaxIteration { get; set; }
        public bool IsConstAlpha { get; set; }
        public long SaveStep { get; set; }
        public int VQ { get; set; }
        public float GradientCutoff { get; set; }

        public void DumpSetting()
        {
            Logger.WriteLine("Use const learning rate: {0}", IsConstAlpha);
            Logger.WriteLine("Starting learning rate: {0}", LearningRate);
            Logger.WriteLine("Max Iteration: {0}", MaxIteration);
            Logger.WriteLine("SIMD: {0}, Size: {1}bits", Vector.IsHardwareAccelerated,
                Vector<float>.Count * sizeof(float) * 8);
            Logger.WriteLine("Gradient cut-off: {0}", GradientCutoff);

            if (SaveStep > 0)
            {
                Logger.WriteLine("Save temporary model after every {0} sentences", SaveStep);
            }
        }
    }
}