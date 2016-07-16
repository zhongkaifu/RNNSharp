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
        public static double GradientCutoff { get; set; }
        public static float LearningRate { get; set; }

        public static Vector<double> vecMaxGrad;
        public static Vector<double> vecMinGrad;
        public static Vector<double> vecNormalLearningRate;

        public static Random rand = new Random(DateTime.Now.Millisecond);

        public static double random(double min, double max)
        {
            return rand.NextDouble() * (max - min) + min;
        }

        public static float RandInitWeight()
        {
            return (float)(random(-0.1, 0.1) + random(-0.1, 0.1) + random(-0.1, 0.1));
        }

        public static double NormalizeGradient(double err)
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
        public static Vector<double> NormalizeGradient(Vector<double> v)
        {
            v = Vector.Min<double>(v, vecMaxGrad);
            v = Vector.Max<double>(v, vecMinGrad);

            return v;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector<double> ComputeLearningRate(Vector<double> vecDelta, ref Vector<double> vecLearningRateWeights)
        {
            vecLearningRateWeights += (vecDelta * vecDelta);
            return vecNormalLearningRate / (Vector<double>.One + Vector.SquareRoot<double>(vecLearningRateWeights));
        }

        public static double UpdateLearningRate(Matrix<double> m, int i, int j, double delta)
        {
            double dg = m[i][j] + delta * delta;
            m[i][j] = dg;

            return LearningRate / (1.0 + Math.Sqrt(dg));
        }

        //Save matrix into file as binary format
        public static void SaveMatrix(Matrix<double> mat, BinaryWriter fo, bool bVQ = false)
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

        public static void matrixXvectorADD(double[] dest, double[] srcvec, Matrix<double> srcmatrix, int DestSize, int SrcSize, bool cleanDest = true)
        {
            Parallel.For(0, DestSize, i =>
            {
                double[] vector_i = srcmatrix[i];
                double cellOutput = 0;
                int j = 0;

                while (j < SrcSize - Vector<double>.Count)
                {
                    Vector<double> v1 = new Vector<double>(srcvec, j);
                    Vector<double> v2 = new Vector<double>(vector_i, j);
                    cellOutput += Vector.Dot<double>(v1, v2);

                    j += Vector<double>.Count;
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


        public static void matrixXvectorADDErr(double[] dest, double[] srcvec, Matrix<double> srcmatrix, int DestSize, int SrcSize)
        {
            Parallel.For(0, DestSize, i =>
            {
                double er = 0;
                for (int j = 0; j < SrcSize; j++)
                {
                    er += srcvec[j] * srcmatrix[j][i];
                }

                dest[i] = RNNHelper.NormalizeGradient(er);
            });
        }

        public static void CheckModelFileType(string filename, out MODELDIRECTION modelDir)
        {
            using (StreamReader sr = new StreamReader(filename))
            {
                BinaryReader br = new BinaryReader(sr.BaseStream);
                int modelType = br.ReadInt32();
                modelDir = (MODELDIRECTION)br.ReadInt32();
            }

            Logger.WriteLine("Get model direction: {0}", modelDir);
        }

        public static Matrix<double> LoadMatrix(BinaryReader br)
        {
            int width = br.ReadInt32();
            int height = br.ReadInt32();
            int vqSize = br.ReadInt32();
            Logger.WriteLine("Loading matrix. width: {0}, height: {1}, vqSize: {2}", width, height, vqSize);

            Matrix<double> m = new Matrix<double>(height, width);
            if (vqSize == 0)
            {
                for (int r = 0; r < height; r++)
                {
                    for (int c = 0; c < width; c++)
                    {
                        m[r][c] = br.ReadDouble();
                    }
                }
            }
            else
            {
                List<double> codeBook = new List<double>();

                for (int i = 0; i < vqSize; i++)
                {
                    codeBook.Add(br.ReadDouble());
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

        public static void matrixXvectorADD(double[] dest, double[] srcvec, Matrix<double> srcmatrix, HashSet<int> setSkipSampling, int SrcSize, bool cleanDest = true)
        {
            Parallel.ForEach(setSkipSampling, i =>
            {
                double cellOutput = 0;
                double[] vector_i = srcmatrix[i];
                int j = 0;
                while (j < SrcSize - Vector<double>.Count)
                {
                    Vector<double> v1 = new Vector<double>(srcvec, j);
                    Vector<double> v2 = new Vector<double>(vector_i, j);
                    cellOutput += Vector.Dot<double>(v1, v2);

                    j += Vector<double>.Count;
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

        public static void matrixXvectorADDErr(double[] dest, double[] srcvec, Matrix<double> srcmatrix, HashSet<int> setSkipSampling, int SrcSize)
        {
            Parallel.ForEach(setSkipSampling, i =>
            {
                double er = 0;
                for (int j = 0; j < SrcSize; j++)
                {
                    er += srcvec[j] * srcmatrix[j][i];
                }

                dest[i] = RNNHelper.NormalizeGradient(er);
            });
        }

        public static void matrixXvectorADDErr(double[] dest, double[] srcvec, Matrix<double> srcmatrix, int DestSize, HashSet<int> setSkipSampling)
        {
            Parallel.For(0, DestSize, i =>
            {
                double er = 0;
                foreach (int j in setSkipSampling)
                {
                    er += srcvec[j] * srcmatrix[j][i];
                }

                dest[i] = RNNHelper.NormalizeGradient(er);
            });
        }
    }
}
