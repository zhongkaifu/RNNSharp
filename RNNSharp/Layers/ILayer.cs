using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RNNSharp.Layers
{
    public interface ILayer
    {
        float[] Cells { get; set; }
        float[] Errs { get; set; }
        int LayerSize { get; }
        LayerType LayerType { get; }
        LayerConfig LayerConfig { get; set; }
        int SparseFeatureSize { get; set; }
        int DenseFeatureSize { get; set; }
        List<SparseVector> SparseFeatureGroups { get; set; }
        List<float[]> DenseFeatureGroups { get; set; }
        ILayer CreateLayerSharedWegiths();
        Neuron CopyNeuronTo(Neuron neuron);
        void SetNeuron(Neuron neuron);
        void InitializeWeights(int sparseFeatureSize, int denseFeatureSize);

        void Save(BinaryWriter fo);
        void Load(BinaryReader br, LayerType layerType, bool forTraining = false);

        void CleanForTraining();
        void ForwardPass(List<SparseVector> sparseFeatureGroups, List<float[]> denseFeatureGroup);
        void ForwardPass(SparseVector sparseFeature, float[] denseFeature);
        void ComputeLayerErr(ILayer prevLayer);
        void ComputeLayerErr(List<float[]> destErrsList, bool cleanDest = true);
        void ComputeLayerErr(float[] destErr, bool cleanDest = true);
        void BackwardPass();
        void UpdateWeights();
        void Reset();
        void SetRunningMode(RunningMode mode);
    }
}
