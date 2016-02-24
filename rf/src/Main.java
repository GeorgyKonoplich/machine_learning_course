import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class Main {
    public final static int N = 100;
    public final static int M = 10000;

    public static double fScore(List<Integer> correct, List<Integer> predicted) {
        int tp = 0;
        int fp = 0;
        int fn = 0;

        int cnt = 0;
        for (int i = 0; i < correct.size(); i++) {
            int o = predicted.get(i);
            int c = correct.get(i);
            if (o == 1 && c == 1) {
                tp++;
            }
            if (o == 1 && c == -1) {
                fp++;
            }
            if (o == -1 && c == 1) {
                fn++;
            }
        }
        double f1 = 2.0 * tp / (2.0 * tp + fp + fn);
        return f1;
    }

    public static void main(String[] args) throws FileNotFoundException {
        Scanner fTrainData = new Scanner(new File("arcene_train.data"));
        Scanner fTrainLabel = new Scanner(new File("arcene_train.labels"));
        Scanner fValidData = new Scanner(new File("arcene_valid.data"));
        Scanner fValidLabel = new Scanner(new File("arcene_valid.labels"));

        List<Item> trainingSet = new ArrayList<Item>();
        System.out.println("> Read train");
        for (int i = 0; i < N; i++) {
            Item item = new Item();
            for (int j = 0; j < M; j++) {
                Integer atr = fTrainData.nextInt();
                String s = "" + j;
                item.setAttribute(s, atr);
            }
            int label = fTrainLabel.nextInt();
            item.setCategory(label);
            trainingSet.add(item);
        }

        System.out.println("ok");
        List<Integer> correctLabels = new ArrayList<Integer>();
        System.out.println("> Read test");
        List<Item> validSet = new ArrayList<>();
        for (int i = 0; i < N; i++) {
            Item item = new Item();
            for (int j = 0; j < M; j++) {
                Integer atr = fValidData.nextInt();
                String s = "" + j;
                item.setAttribute(s, atr);
            }
            int label = fValidLabel.nextInt();
            item.setCategory(label);
            correctLabels.add(label);
            validSet.add(item);
        }
        System.out.println("ok");

        DecisionTreeBuilder tbuilder =
                DecisionTree
                        .createBuilder()
                        .setTrainingSet(trainingSet)
                        .setDefaultPredicates(Predicate.GTE, Predicate.LTE);

        //DecisionTree tree = tbuilder.createDecisionTree();
        //System.out.println("Decision tree created");

        RandomForest forest = tbuilder.createRandomForest(15);
        System.out.println("Random forest created");
        List<Integer> predictedLabels = new ArrayList<Integer>();

        for (int i = 0; i < N;i++){
            Map<Object, Integer> result = forest.classify(validSet.get(i));
            int red = 0;
            int green = 0;
            for (Object color : result.keySet()) {
                if (color.equals(-1)) {
                    red = result.get(color);
                } else if (color.equals(1)) {
                    green = result.get(color);
                }
            }
            int ctg = 0;
            if (red >= green) ctg = -1; else ctg = 1;
            predictedLabels.add(ctg);
        }

        System.out.println(fScore(correctLabels, predictedLabels));
    }


}
