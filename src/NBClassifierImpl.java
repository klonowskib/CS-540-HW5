/***************************************************************************************
 CS540 - Section 2
 Homework Assignment 5: Naive Bayes

 NBClassifierImpl.java
 This is the main class that implements functions for Naive Bayes Algorithm!
 ---------
 *Free to modify anything in this file, except the class name
 You are required:
 - To keep the class name as NBClassifierImpl for testing
 - Not to import any external libraries
 - Not to include any packages
 *Notice: To use this file, you should implement 2 methods below.

 @author: TA
 @date: April 2017
 *****************************************************************************************/

import java.util.ArrayList;
import java.util.List;


public class NBClassifierImpl implements NBClassifier {

	private int nFeatures;        // The number of features including the class
	private int[] featureSize;    // Size of each features
	private List<List<Double[]>> logPosProbs;    // parameters of Naive Bayes

	/**
	 * Constructs a new classifier without any trained knowledge.
	 */
	public NBClassifierImpl() {

	}

	/**
	 * Construct a new classifier
	 *
	 * @param features int[] sizes of all attributes
	 */
	public NBClassifierImpl(int[] features) {
		this.nFeatures = features.length;

		// initialize feature size
		this.featureSize = features.clone();

		this.logPosProbs = new ArrayList<List<Double[]>>(this.nFeatures);
	}


	/**
	 * Read training data and learn parameters
	 *
	 * @param data training data
	 */
	@Override
	public void fit(int[][] data) {
		int pos_count = 0;
		for (int[] inst : data) {
			if (inst[inst.length-1]==1)
				pos_count++;
		}
		List<Double[]> pos = new ArrayList<>(); //Hold the probabilities for a positive or negative classification
		Double[] y_pos = new Double[2];
		Double[] y_neg = new Double[2];
		y_neg[0] = Math.log((data.length - pos_count + 1)/((double)data.length + featureSize[featureSize.length - 1])); // probability that the class is negative
		y_neg[1] = 0.0;
		y_pos[0] = 0.0;
		y_pos[1] = Math.log(((double)pos_count+1) / ((double)data.length + featureSize[featureSize.length - 1])); // probability that the class is positive
		pos.add(0, y_neg);
		pos.add(1, y_pos);
		//logPosProbs.add(logPosProbs.size()- 1, pos);
		for (int i = 0; i < data[0].length - 1; i++) { //for each attribute
			List<Double[]> attribute_values = new ArrayList<>(); //Will hold the probabilities of the attribute values
			for (int j = 0; j < featureSize[i]; j++) {//for each value that the feature can take
				Double[] attr = new Double[2]; //An array that holds the probability of the value, given positive and negative classes
				int p_j_given_pos;
				int joint = 0; //counts instances where the attribute takes the given value and class is positive
				int n_joint = 0; //counts instances where the attribute takes the given value and class is negative
				for (int k = 0; k < data.length; k++) { //for each instance
					if(data[k][i] == j) { //if the value matches
						if(data[k][data[k].length-1] == 1) //and the class is positive
							joint++;
						else if (data[k][data[k].length-1] == 0)//and the class is negative
							n_joint++;
					}
				}
				attr[0] = Math.log(((double)n_joint + 1)/((data.length-pos_count) + featureSize[i])); //conditional probability with a negative classification
				attr[1] = Math.log(((double)joint + 1)/(pos_count + featureSize[i])); //conditional probability with a positive classification
				attribute_values.add(attr);
			}
			logPosProbs.add(attribute_values);
		}
		logPosProbs.add(pos);
	}

	/**
	 * Classify new dataset
	 *
	 * @param instances int[][] test data
	 * @return Label[] classified labels
	 */
	@Override
	public Label[] classify(int[][] instances) {
		int nrows = instances.length;
		Label[] yPred = new Label[nrows]; // predicted data
		int i = 0;
		for(int[] inst : instances) {
			yPred[i] = Label.Positive;
			double cond_positive = 0;
			double cond_negative = 0;
			for (int attr_ind = 0; attr_ind < nFeatures-1; attr_ind++) {
				int actual = inst[attr_ind];
				cond_negative += logPosProbs.get(attr_ind).get(actual)[0];
				cond_positive += logPosProbs.get(attr_ind).get(actual)[1];
			}
			cond_positive += logPosProbs.get(nFeatures-1).get(1)[1];
			cond_negative += logPosProbs.get(nFeatures-1).get(0)[0];
			if(cond_negative > cond_positive)
				yPred[i] = Label.Negative;
			i++;
		}
		return yPred;
	}
}