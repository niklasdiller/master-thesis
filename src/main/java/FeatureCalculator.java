package main.java;

import com.google.common.math.IntMath;
import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.descriptive.moment.Kurtosis;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.moment.Skewness;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.apache.commons.math3.stat.descriptive.rank.Median;
import org.apache.commons.math3.transform.DftNormalization;
import org.apache.commons.math3.transform.FastFourierTransformer;
import org.apache.commons.math3.transform.TransformType;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class FeatureCalculator {

  public double min(double[] values) throws Exception {
    return Arrays.stream(values).min().orElseThrow(() -> new Exception("Empty value array given"));
  }

  public double max(double[] values) throws Exception {
    return Arrays.stream(values).max().orElseThrow(() -> new Exception("Empty value array given"));
  }

  public double mag(double[] values) {
    RealVector v = MatrixUtils.createRealVector(values);
    return v.getNorm();
  }

  public double mean(double[] values) {
    return new Mean().evaluate(values);
  }

  public double median(double[] values) {
    return new Median().evaluate(values);
  }

  public double std(double[] values) {
    return new StandardDeviation().evaluate(values);
  }

  public double iqr(double[] values) {
    DescriptiveStatistics da = new DescriptiveStatistics(values);
    return da.getPercentile(75) - da.getPercentile(25);
  }

  public double skew(double[] values) {
    return new Skewness().evaluate(values);
  }

  public double kurtosis(double[] values) {
    return new Kurtosis().evaluate(values);
  }

  public double rms(double[] values) {
    double sum = 0;
    for (double d : values) {
      sum = sum + d * d;
    }
    return Math.sqrt(sum / values.length);
  }

  public double mcr(double[] values) {
    double[] meanArr = new double[values.length];
    double mean = mean(values);

    for (int i = 0; i < values.length; i++) {
      meanArr[i] = values[i] - mean;
    }

    int sum = 0;
    for (int i = 0; i < meanArr.length - 1; i++) {
      if (meanArr[i] * meanArr[i + 1] < 0)
        sum = sum + 1;
    }
    return (double) sum / values.length;
  }

  public double energy(double[] values) {
    double[] meanArr = new double[(IntMath.isPowerOfTwo(values.length)) ? values.length
            : IntMath.ceilingPowerOfTwo(values.length)];

    double mean = mean(values);
    for (int i = 0; i < values.length; i++) {
      meanArr[i] = values[i] - mean;
    }

    if (values.length != meanArr.length) {
      for (int i = values.length; i < meanArr.length; i++)
        meanArr[i] = 0.0;
    }

    // TODO: Check if UNITARY and FORWARD are correct
    FastFourierTransformer fft = new FastFourierTransformer(DftNormalization.UNITARY);
    Complex[] result = fft.transform(meanArr, TransformType.FORWARD);
    double sum = 0;
    for (int i = 0; i < result.length; i++) {
      sum = sum + result[i].abs();
    }

    return sum / values.length;
  }

  public double peakFreq(double[] values) {
    double[] meanArr = new double[(IntMath.isPowerOfTwo(values.length))?
            values.length:IntMath.ceilingPowerOfTwo(values.length)];
    double mean = mean(values);
    for(int i = 0; i < values.length; i++) {
      meanArr[i] = values[i]-mean;
    }

    if(values.length != meanArr.length) {
      for(int i = values.length; i < meanArr.length; i++)
        meanArr[i] = 0.0;
    }
    // TODO: Check if UNITARY and FORWARD are correct
    FastFourierTransformer fft = new FastFourierTransformer(DftNormalization.UNITARY);
    Complex[] result = fft.transform(meanArr, TransformType.FORWARD);
    double[] magArr = new double[result.length];
    for(int i = 0; i < result.length; i++) {
      magArr[i] = result[i].abs() / values.length;

    }
    double[] lnSpace = this.linspace(0, 5, magArr.length);
    List<Double> list = new ArrayList<>(magArr.length);
    for (double v: magArr) {
      list.add(v);
    }
    Arrays.sort(magArr);
    int index = list.indexOf(Double.valueOf(magArr[magArr.length - 1]));
    return lnSpace[index];
  }

  public double frDmEntropy(double[] values) {
    double[] meanArr = new double[(IntMath.isPowerOfTwo(values.length)) ? values.length
            : IntMath.ceilingPowerOfTwo(values.length)];
    double[] probabilityArr = new double[(IntMath.isPowerOfTwo(values.length)) ? values.length
            : IntMath.ceilingPowerOfTwo(values.length)];
    double[] magArray = new double[(IntMath.isPowerOfTwo(values.length)) ? values.length
            : IntMath.ceilingPowerOfTwo(values.length)];

    double mean = mean(values);
    for (int i = 0; i < values.length; i++) {
      meanArr[i] = values[i] - mean;
    }
    if (values.length != meanArr.length) {
      for (int i = values.length; i < meanArr.length; i++)
        meanArr[i] = 0.0;
    }
    // TODO: Check if UNITARY and FORWARD are correct
    FastFourierTransformer fft = new FastFourierTransformer(DftNormalization.UNITARY);
    Complex[] result = fft.transform(meanArr, TransformType.FORWARD);

    for (int i = 0; i < result.length; i++) {
      magArray[i] = result[i].abs() / values.length;
    }
    for (int i = 0; i < result.length; i++) {
      probabilityArr[i] = magArray[i] / Arrays.stream(magArray).sum();
    }

    Double frDmEntroPy = calcEntropy(probabilityArr);
    if(frDmEntroPy.isNaN()) {
      frDmEntroPy = 0.0;
    }
    return frDmEntroPy;
  }

  private double[] linspace(double min, double max, int points) {
    double[] d = new double[points];
    for (int i = 0; i < points; i++){
      d[i] = min + i * (max - min) / (points - 1);
    }
    return d;
  }

  private double calcEntropy(double[] probabilityArr) {
    double entropy = 0.0;
    for (int i = 0; i < probabilityArr.length; i++) {
      entropy = entropy + (-(probabilityArr[i]) * ((int) (Math.log(probabilityArr[i]) / Math.log(2))));
    }
    return entropy;
  }

  public static void main(String[] args) throws Exception {
    FeatureCalculator calc = new FeatureCalculator();

    double[] values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    System.out.println("Max: " + calc.max(values));
    System.out.println("Min: " + calc.min(values));
    System.out.println("Mag: " + calc.mag(values));
    System.out.println("Mean: " + calc.mean(values));
    System.out.println("Median: " + calc.median(values));
    System.out.println("STD: " + calc.std(values));
    System.out.println("IQR: " + calc.iqr(values));
    System.out.println("Skew: " + calc.skew(values));
    System.out.println("Kurtosis: " + calc.kurtosis(values));
    System.out.println("RMS: " + calc.rms(values));
    System.out.println("MCR: " + calc.mcr(values));
    System.out.println("Energy: " + calc.energy(values));
    System.out.println("PeakFreq: " + calc.peakFreq(values));
    System.out.println("Entropy: " + calc.frDmEntropy(values));
  }
}
