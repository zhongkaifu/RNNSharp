using AdvUtils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace RNNSharp
{
    public class RNNHelper
    {
        public static float GradientCutoff { get; set; }
        public static float LearningRate { get; set; }
        public static bool IsConstAlpha { get; set; }

        public static Vector<float> vecMaxGrad;
        public static Vector<float> vecMinGrad;
        public static Vector<float> vecNormalLearningRate;
        public static Random rand = new Random(DateTime.Now.Millisecond);

        public static float random(float min, float max)
        {
            return (float)(rand.NextDouble() * (max - min) + min);
        }

        public static float RandInitWeight()
        {
            return (float)(random(-0.1f, 0.1f) + random(-0.1f, 0.1f) + random(-0.1f, 0.1f));
        }

        public static float NormalizeGradient(float err)
        {
            if (err > GradientCutoff)
            {
                err = GradientCutoff;
            }
            else if (err < -GradientCutoff)
            {
                err = -GradientCutoff;
            }
            return err;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector<float> NormalizeGradient(Vector<float> v)
        {
            v = Vector.Min<float>(v, vecMaxGrad);
            v = Vector.Max<float>(v, vecMinGrad);

            return v;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void UpdateFeatureWeights(float[] feature, float[] featureWeight, float[] learningRateWeight, float err, int idx)
        {
            //Computing error delta
            Vector<float> vecDenseFeature = new Vector<float>(feature, idx);
            Vector<float> vecDelta = vecDenseFeature * err;
 
            vecDelta = NormalizeGradient(vecDelta);

            //Computing learning rate
            Vector<float> vecDenseWeightLearningRateCol = new Vector<float>(learningRateWeight, idx);
            vecDenseWeightLearningRateCol += vecDelta * vecDelta;
            vecDenseWeightLearningRateCol.CopyTo(learningRateWeight, idx);

            Vector<float> vecNewLearningRate = vecNormalLearningRate / (Vector<float>.One + Vector.SquareRoot<float>(vecDenseWeightLearningRateCol));

            Vector<float> vecVector_C = new Vector<float>(featureWeight, idx);
            vecVector_C += vecNewLearningRate * vecDelta;
            vecVector_C.CopyTo(featureWeight, idx);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector<float> UpdateLearningRate(Vector<float> vecDelta, ref Vector<float> vecLearningRateWeights)
        {
            if (IsConstAlpha)
            {
                return vecNormalLearningRate;
            }
            else
            {
                vecLearningRateWeights += (vecDelta * vecDelta);
                return vecNormalLearningRate / (Vector<float>.One + Vector.SquareRoot<float>(vecLearningRateWeights));
            }
        }

        public static float UpdateLearningRate(Matrix<float> m, int i, int j, float delta)
        {
            if (IsConstAlpha)
            {
                return LearningRate;
            }
            else
            {
                float dg = m[i][j] + delta * delta;
                m[i][j] = dg;

                return (float)(LearningRate / (1.0 + Math.Sqrt(dg)));
            }
        }

        //Save matrix into file as binary format
        public static void SaveMatrix(Matrix<float> mat, BinaryWriter fo, bool bVQ = false)
        {
            //Save the width and height of the matrix
            fo.Write(mat.Width);
            fo.Write(mat.Height);

            if (bVQ == false)
            {
                Logger.WriteLine("Saving matrix without VQ...");
                fo.Write(0); // non-VQ

                //Save the data in matrix
                for (int r = 0; r < mat.Height; r++)
                {
                    for (int c = 0; c < mat.Width; c++)
                    {
                        fo.Write(mat[r][c]);
                    }
                }
            }
            else
            {
                //Build vector quantization matrix
                int vqSize = 256;
                VectorQuantization vq = new VectorQuantization();
                Logger.WriteLine("Saving matrix with VQ {0}...", vqSize);

                int valSize = 0;
                for (int i = 0; i < mat.Height; i++)
                {
                    for (int j = 0; j < mat.Width; j++)
                    {
                        vq.Add(mat[i][j]);
                        valSize++;
                    }
                }

                if (vqSize > valSize)
                {
                    vqSize = valSize;
                }

                double distortion = vq.BuildCodebook(vqSize);
                Logger.WriteLine("Distortion: {0}, vqSize: {1}", distortion, vqSize);

                //Save VQ codebook into file
                fo.Write(vqSize);
                for (int j = 0; j < vqSize; j++)
                {
                    fo.Write(vq.CodeBook[j]);
                }

                //Save the data in matrix
                for (int r = 0; r < mat.Height; r++)
                {
                    for (int c = 0; c < mat.Width; c++)
                    {
                        fo.Write((byte)vq.ComputeVQ(mat[r][c]));
                    }
                }
            }
        }
        public static float[] ConcatenateVector(VectorBase src1, float[] src2)
        {
            float[] dest = new float[src1.Length + src2.Length];
            Parallel.Invoke(() =>
            {
                src1.CopyTo().CopyTo(dest, 0);
            }
            ,
             () =>
             {
                 src2.CopyTo(dest, src1.Length);
             });

            return dest;
        }

        public static float[] ConcatenateVector(float[] src1, float[] src2)
        {
            float[] dest = new float[src1.Length + src2.Length];
            Parallel.Invoke(() =>
            {
                src1.CopyTo(dest, 0);
            }
            ,
             () =>
             {
                 src2.CopyTo(dest, src1.Length);
             });

            return dest;
        }

        public static void matrixXvectorADD(float[] dest, float[] srcvec, Matrix<float> srcmatrix, int DestSize, int SrcSize, bool cleanDest = true)
        {
            Parallel.For(0, DestSize, i =>
            {
                float[] vector_i = srcmatrix[i];
                float cellOutput = 0;
                int j = 0;

                while (j < SrcSize - Vector<float>.Count)
                {
                    Vector<float> v1 = new Vector<float>(srcvec, j);
                    Vector<float> v2 = new Vector<float>(vector_i, j);
                    cellOutput += Vector.Dot<float>(v1, v2);

                    j += Vector<float>.Count;
                }

                while (j < SrcSize)
                {
                    cellOutput += srcvec[j] * vector_i[j];
                    j++;
                }

                if (cleanDest)
                {
                    dest[i] = cellOutput;
                }
                else
                {
                    dest[i] += cellOutput;
                }
            });
        }


        public static void matrixXvectorADDErr(float[] dest, float[] srcvec, Matrix<float> srcmatrix, int DestSize, int SrcSize)
        {
            Parallel.For(0, DestSize, i =>
            {
                float er = 0;
                for (int j = 0; j < SrcSize; j++)
                {
                    er += srcvec[j] * srcmatrix[j][i];
                }

                dest[i] = RNNHelper.NormalizeGradient(er);
            });
        }

        public static Matrix<float> LoadMatrix(BinaryReader br)
        {
            int width = br.ReadInt32();
            int height = br.ReadInt32();
            int vqSize = br.ReadInt32();
            Logger.WriteLine("Loading matrix. width: {0}, height: {1}, vqSize: {2}", width, height, vqSize);

            Matrix<float> m = new Matrix<float>(height, width);
            if (vqSize == 0)
            {
                for (int r = 0; r < height; r++)
                {
                    for (int c = 0; c < width; c++)
                    {
                        m[r][c] = br.ReadSingle();
                    }
                }
            }
            else
            {
                List<float> codeBook = new List<float>();

                for (int i = 0; i < vqSize; i++)
                {
                    codeBook.Add(br.ReadSingle());
                }


                for (int r = 0; r < height; r++)
                {
                    for (int c = 0; c < width; c++)
                    {
                        int vqIndex = br.ReadByte();
                        m[r][c] = codeBook[vqIndex];
                    }
                }
            }

            return m;
        }

        public static void matrixXvectorADD(float[] dest, float[] srcvec, Matrix<float> srcmatrix, HashSet<int> setSkipSampling, int SrcSize, bool cleanDest = true)
        {
            Parallel.ForEach(setSkipSampling, i =>
            {
                float cellOutput = 0;
                float[] vector_i = srcmatrix[i];
                int j = 0;
                while (j < SrcSize - Vector<float>.Count)
                {
                    Vector<float> v1 = new Vector<float>(srcvec, j);
                    Vector<float> v2 = new Vector<float>(vector_i, j);
                    cellOutput += Vector.Dot<float>(v1, v2);

                    j += Vector<float>.Count;
                }

                while (j < SrcSize)
                {
                    cellOutput += srcvec[j] * vector_i[j];
                    j++;
                }

                if (cleanDest)
                {
                    dest[i] = cellOutput;
                }
                else
                {
                    dest[i] += cellOutput;
                }
            });

        }

        public static void matrixXvectorADDErr(float[] dest, float[] srcvec, Matrix<float> srcmatrix, HashSet<int> setSkipSampling, int SrcSize)
        {
            Parallel.ForEach(setSkipSampling, i =>
            {
                float er = 0;
                for (int j = 0; j < SrcSize; j++)
                {
                    er += srcvec[j] * srcmatrix[j][i];
                }

                dest[i] = RNNHelper.NormalizeGradient(er);
            });
        }

        public static void matrixXvectorADDErr(float[] dest, float[] srcvec, Matrix<float> srcmatrix, int DestSize, HashSet<int> setSkipSampling)
        {
            Parallel.For(0, DestSize, i =>
            {
                float er = 0;
                foreach (int j in setSkipSampling)
                {
                    er += srcvec[j] * srcmatrix[j][i];
                }

                dest[i] = RNNHelper.NormalizeGradient(er);
            });
        }
    }
}
