package pca;

import Jama.Matrix;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;

public class Main {
    public static void main(String[] args) throws FileNotFoundException {
        Scanner sc = new Scanner(new File(args[0]));
        List<Double[]> list = new ArrayList<>();

        while (sc.hasNextLine()) {
            String[] strings = sc.nextLine().split(" ");
            Double[] features = new Double[strings.length];
            for (int i = 0; i < strings.length; i++) features[i] = Double.parseDouble(strings[i]);
            list.add(features);
        }

        int n = list.size();
        int m = list.get(0).length;
        double[][] f = new double[n][m];

        for (int i = 0; i < n; i++){
            for (int j = 0; j < m; j++) f[i][j] = list.get(i)[j];
        }

        double[] u = new double[m];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                u[i] += f[j][i];
            }
            u[i] /= n;
        }

        double[][] b = new double[n][m];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                b[i][j] = f[i][j] - u[j];
            }
        }


        Matrix g = new Matrix(b);
        Matrix gTranspose = g.transpose().copy();
        Matrix p = gTranspose.times(g);
        p.timesEquals(1.0 / (m - 1));
        Matrix eqw = p.eig().getD();

        double[] eigenvalues = p.eig().getRealEigenvalues();

        for(int i = 0;i < eigenvalues.length;++i){
            eigenvalues[i] = eqw.get(i,i);
        }

        double acc = Double.parseDouble(args[1]);
        double sum = 0;
        Arrays.sort(eigenvalues);

        for (int i = 0; i < eigenvalues.length; ++i) {
            sum += eigenvalues[i];
        }

        double part = 0;

        for (int i = eigenvalues.length - 1; i >= 0; i--) {
            part += eigenvalues[i];
            if(part/sum > acc){
                System.out.println((eigenvalues.length - i) + "   " + part/sum);
                return;
            }
        }
        System.out.println("error");
    }
}
