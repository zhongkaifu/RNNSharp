namespace RNNSharp
{
    public interface ISequence
    {
        int DenseFeatureSize { get; }
        int SparseFeatureSize { get; }
    }
}