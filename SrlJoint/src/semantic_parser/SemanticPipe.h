// Copyright (c) 2012-2015 Andre Martins
// All Rights Reserved.
//
// This file is part of TurboParser 2.3.
//
// TurboParser 2.3 is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// TurboParser 2.3 is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with TurboParser 2.3.  If not, see <http://www.gnu.org/licenses/>.

#ifndef SemanticPipe_H_
#define SemanticPipe_H_

#include "Pipe.h"
#include "SemanticOptions.h"
#include "SemanticReader.h"
#include "SemanticDictionary.h"
#include "TokenDictionary.h"
#include "SemanticInstanceNumeric.h"
#include "SemanticWriter.h"
#include "SemanticPart.h"
#include "SemanticDecoder.h"
#include "Parser.h"
#include "Pruner.h"
#include "dynet/io.h"
#include "dynet/mp.h"

class SemanticPipe : public Pipe {
public:
    SemanticPipe(Options *semantic_options) : Pipe(semantic_options) {
        token_dictionary_ = NULL;
        trainer_ = NULL;
        pruner_trainer_ = NULL;
        model_ = NULL;
        pruner_model_ = NULL;
        parser_ = NULL;
        pruner_ = NULL;
    }

    virtual ~SemanticPipe() {
        delete token_dictionary_;
        delete trainer_;
        delete parser_;
        delete model_;
        if (pruner_trainer_) delete pruner_trainer_;
        if (pruner_model_) delete pruner_model_;
        if (pruner_) delete pruner_;
    }

    SemanticReader *GetSemanticReader() {
        return static_cast<SemanticReader *>(reader_);
    }

    SemanticDictionary *GetSemanticDictionary() {
        return static_cast<SemanticDictionary *>(dictionary_);
    }

    SemanticDecoder *GetSemanticDecoder() {
        return static_cast<SemanticDecoder *>(decoder_);
    }

    SemanticOptions *GetSemanticOptions() {
        return static_cast<SemanticOptions *>(options_);
    }

    void Initialize();

    void LoadPretrainedEmbedding(bool load_parser_embedding, bool load_pruner_embedding);

    void Train();

    void TrainPruner();

    double TrainEpoch(const vector<int> &idxs, const vector<int> &exemplar_idxs,
                      int epoch, double &best_F1);

    double TrainPrunerEpoch(const vector<int> &idxs, int epoch);

    void Test();

    void Run(double &F1);

    void RunPruner(double &accuracy);

    void LoadNeuralModel();

    void SaveNeuralModel();

    void LoadPruner();

    void SavePruner();

    void BuildFormCount();

protected:
    void CreateDictionary() {
        dictionary_ = new SemanticDictionary(this);
        GetSemanticDictionary()->SetTokenDictionary(token_dictionary_);
    }

    void CreateReader() {
        reader_ = new SemanticReader(options_);
    }

    void CreateWriter() {
        writer_ = new SemanticWriter(options_);
    }

    void CreateDecoder() { decoder_ = new SemanticDecoder(this); }

    Parts *CreateParts() { return new SemanticParts; }

    void CreateTokenDictionary() {
        token_dictionary_ = new TokenDictionary(this);
    }

    void PreprocessData();

	void DeleteInstances() {
		for (int i = 0; i < instances_.size(); ++i) {
			delete instances_[i];
		}
		instances_.clear();

		for (int i = 0; i < dev_instances_.size(); ++i) {
			delete dev_instances_[i];
		}
		dev_instances_.clear();

		for (int i = 0; i < exemplar_instances_.size(); ++i) {
			delete exemplar_instances_[i];
		}
		exemplar_instances_.clear();

	}

    void CreateInstances() {
        timeval start, end;
        gettimeofday(&start, NULL);

        LOG(INFO) << "Creating instances...";

        reader_->Open(options_->GetTrainingFilePath());
        DeleteInstances();
        Instance *instance = reader_->GetNext();
        while (instance) {
            AddInstance(instance);
            instance = reader_->GetNext();
        }
        reader_->Close();

	    reader_->Open(options_->GetTestFilePath());
	    instance = reader_->GetNext();
	    while (instance) {
		    dev_instances_.push_back(instance);
		    instance = reader_->GetNext();
	    }
	    reader_->Close();


        if (GetSemanticOptions()->use_exemplar()) {
            reader_->Open(GetSemanticOptions()->GetExemplarFilePath());
            instance = reader_->GetNext();
            while (instance) {
	            Instance *formatted_instance = GetFormattedInstance(instance);
	            exemplar_instances_.push_back(formatted_instance);
	            if (instance != formatted_instance) delete instance;
                instance = reader_->GetNext();
            }
            reader_->Close();
        }

        LOG(INFO) << "Number of instances: " << instances_.size();

        gettimeofday(&end, NULL);
        LOG(INFO) << "Time: " << diff_ms(end, start);
    }

    Instance *GetFormattedInstance(Instance *instance) {
        SemanticInstanceNumeric *instance_numeric =
                new SemanticInstanceNumeric;
        instance_numeric->Initialize(*GetSemanticDictionary(), static_cast<SemanticInstance *>(instance));
        return instance_numeric;
    }

    void SaveModel(FILE *fs);

    void LoadModel(FILE *fs);

    void MakeParts(Instance *instance, Parts *parts, vector<double> *gold_outputs, int t);

    void MakePartsBasic(Instance *instance, Parts *parts, 
                        vector<double> *gold_outputs, bool is_pruner, int t);

    void MakePartsLabeled(Instance *instance, Parts *parts, vector<double> *gold_outputs, int t);

    void LabelInstance(Parts *parts, const vector<double> &gold_output, const vector<double> &predicted_output,
                       Instance *gold_instance, Instance *predicted_instance);

    virtual void BeginEvaluation() {
        num_predicted_arguments_ = 0;
        num_gold_arguments_ = 0;
        num_matched_arguments_ = 0;
        num_pruned_gold_arguments_ = 0;
        npa_ = nma_ = nga_ = 0.0;
        num_matched_frames_ = num_gold_frames_ = 0.0;
        gettimeofday(&start_clock_, NULL);
    }

    void EvaluateInstance(Instance *instance, Instance *gold_instance, Instance *predicted_instance);

    virtual void EndEvaluation(double &F1) {
        double precision =
                static_cast<double>(num_matched_arguments_) /
                static_cast<double>(num_predicted_arguments_);
        double recall =
                static_cast<double>(num_matched_arguments_) /
                static_cast<double>(num_gold_arguments_);
        if (num_matched_arguments_ > 0)
            F1 = 2.0 * precision * recall /  (precision + recall);
        else
            F1 = 0.;

        double o_p = nma_ / npa_;
        double o_r = nma_ / nga_;
        double o_f = 0.0;
        if (nma_ > 0) o_f = 2.0 * o_p * o_r / (o_p + o_r);
        double pruning_recall = 1.0 * (num_gold_arguments_ - num_pruned_gold_arguments_) / num_gold_arguments_;
        LOG(INFO) << "Pruning recall: "<< pruning_recall;
        LOG(INFO) << "Precision: " << precision
                  << " (" << num_matched_arguments_ << "/"
                  << num_predicted_arguments_ << ")" << " recall: " << recall
                  << " (" << num_matched_arguments_ << "/"
                  << num_gold_arguments_ << ")" << " F1: " << F1;
        LOG(INFO) << "Official precision: " << o_p
                  << " official recall: " << o_r
                   << " official F1: " << o_f;
        LOG(INFO) << "Frame id acc: " << num_matched_frames_ / num_gold_frames_;
        F1 = o_f;
    }

    /* Virtual function from Pipe.h but not implemented. */
    void ComputeScores(Instance *instance, Parts *parts, Features *features,
                       vector<double> *scores) {
        CHECK(false) << "Not implemented." << endl;
    }

    void RemoveUnsupportedFeatures(Instance *instance, Parts *parts,
                                   const vector<bool> &selected_parts,
                                   Features *features) {
        CHECK(false) << "Not implemented." << endl;
    }

    void MakeFeatureDifference(Parts *parts,
                               Features *features,
                               const vector<double> &gold_output,
                               const vector<double> &predicted_output,
                               FeatureVector *difference) {
        CHECK(false) << "Not implemented." << endl;
    }

    void MakeGradientStep(Parts *parts,
                          Features *features,
                          double eta,
                          int iteration,
                          const vector<double> &gold_output,
                          const vector<double> &predicted_output) {
        CHECK(false) << "Not implemented." << endl;
    }

    void TouchParameters(Parts *parts, Features *features,
                         const vector<bool> &selected_parts) {
        CHECK(false) << "Not implemented." << endl;
    }

    void MakeSelectedFeatures(Instance *instance, Parts *parts,
                              const vector<bool> &selected_parts, Features *features) {
        CHECK(false) << "Not implemented." << endl;
    }


    Features *CreateFeatures() { CHECK(false) << "Not implemented." << endl; }


    void MakeParts(Instance *instance, Parts *parts,
                   vector<double> *gold_outputs) {
        CHECK(false) << "Not implemented." << endl;
    }

    void LabelInstance(Parts *parts, const vector<double> &output,
                       Instance *instance) { CHECK(false) << "Not implemented." << endl; }

public:
    ParameterCollection *model_;
    ParameterCollection *pruner_model_;
    Parser *parser_;
    Pruner *pruner_;
    Trainer *trainer_;
    Trainer *pruner_trainer_;
protected:
    vector<Instance *> dev_instances_;
    TokenDictionary *token_dictionary_;
    int num_predicted_arguments_;
    int num_matched_arguments_;
    int num_gold_arguments_;
    int num_pruned_gold_arguments_;
    double npa_;
    double nma_;
    double nga_;
    double num_matched_frames_;
    double num_gold_frames_;
    vector<Instance *> exemplar_instances_;

    timeval start_clock_;
    unordered_map<int, vector<float>> *embedding_;
    unordered_map<int, int> *form_count_;
};

#endif /* SemanticPipe_H_ */
