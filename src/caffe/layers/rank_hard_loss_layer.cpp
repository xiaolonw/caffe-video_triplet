#include <vector>

#include <algorithm>
#include <cmath>
#include <cfloat>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

using std::max;

using namespace std;
using namespace cv;

namespace caffe {

int myrandom (int i) { return caffe_rng_rand()%i;}


template <typename Dtype>
void RankHardLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);

  diff_.ReshapeLike(*bottom[0]);
  dis_.Reshape(bottom[0]->num(), bottom[0]->num(), 1, 1);
  mask_.Reshape(bottom[0]->num(), bottom[0]->num(), 1, 1);
}


template <typename Dtype>
void RankHardLossLayer<Dtype>::set_mask(const vector<Blob<Dtype>*>& bottom)
{

	RankParameter rank_param = this->layer_param_.rank_param();
	int neg_num = rank_param.neg_num();
	int pair_size = rank_param.pair_size();
	float hard_ratio = rank_param.hard_ratio();
	float rand_ratio = rank_param.rand_ratio();
	float margin = rank_param.margin();

	int hard_num = neg_num * hard_ratio;
	int rand_num = neg_num * rand_ratio;

	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* label = bottom[1]->cpu_data();
	int count = bottom[0]->count();
	int num = bottom[0]->num();
	int dim = bottom[0]->count() / bottom[0]->num();
	Dtype* dis_data = dis_.mutable_cpu_data();
	Dtype* mask_data = mask_.mutable_cpu_data();

	for(int i = 0; i < num * num; i ++)
	{
		dis_data[i] = 0;
		mask_data[i] = 0;
	}

	// calculate distance
	for(int i = 0; i < num; i ++)
	{
		for(int j = i + 1; j < num; j ++)
		{
			const Dtype* fea1 = bottom_data + i * dim;
			const Dtype* fea2 = bottom_data + j * dim;
			Dtype ts = 0;
			for(int k = 0; k < dim; k ++)
			{
			  ts += (fea1[k] * fea2[k]) ;
			}
			dis_data[i * num + j] = -ts;
			dis_data[j * num + i] = -ts;
		}
	}

	//select samples

	vector<pair<float, int> >negpairs;
	vector<int> sid1;
	vector<int> sid2;


	for(int i = 0; i < num; i += pair_size)
	{
		negpairs.clear();
		sid1.clear();
		sid2.clear();
		for(int j = 0; j < num; j ++)
		{
			if(label[j] == label[i])
				continue;
			Dtype tloss = max(Dtype(0), dis_data[i * num + i + 1] - dis_data[i * num + j] + Dtype(margin));
			if(tloss == 0) continue;

			negpairs.push_back(make_pair(dis_data[i * num + j], j));
		}
		if(negpairs.size() <= neg_num)
		{
			for(int j = 0; j < negpairs.size(); j ++)
			{
				int id = negpairs[j].second;
				mask_data[i * num + id] = 1;
			}
			continue;
		}
		sort(negpairs.begin(), negpairs.end());

		for(int j = 0; j < neg_num; j ++)
		{
			sid1.push_back(negpairs[j].second);
		}
		for(int j = neg_num; j < negpairs.size(); j ++)
		{
			sid2.push_back(negpairs[j].second);
		}
		std::random_shuffle(sid1.begin(), sid1.end(), myrandom);
		for(int j = 0; j < min(hard_num, (int)(sid1.size()) ); j ++)
		{
			mask_data[i * num + sid1[j]] = 1;
		}
		for(int j = hard_num; j < sid1.size(); j ++)
		{
			sid2.push_back(sid1[j]);
		}
		std::random_shuffle(sid2.begin(), sid2.end(), myrandom);
		for(int j = 0; j < min( rand_num, (int)(sid2.size()) ); j ++)
		{
			mask_data[i * num + sid2[j]] = 1;
		}

	}


}




template <typename Dtype>
void RankHardLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* label = bottom[1]->cpu_data();
	int count = bottom[0]->count();
	int num = bottom[0]->num();
	int dim = bottom[0]->count() / bottom[0]->num();


	RankParameter rank_param = this->layer_param_.rank_param();
	int neg_num = rank_param.neg_num();      // 4
	int pair_size = rank_param.pair_size();  // 5
	float hard_ratio = rank_param.hard_ratio();
	float rand_ratio = rank_param.rand_ratio();
	float margin = rank_param.margin();
	Dtype* dis_data = dis_.mutable_cpu_data();
	Dtype* mask_data = mask_.mutable_cpu_data();

	set_mask(bottom);
	Dtype loss = 0;
	int cnt = neg_num * num / pair_size * 2;

	for(int i = 0; i < num; i += pair_size)
	{
		for(int j = 0; j < num; j ++)
		{
			if(mask_data[i * num + j] == 0) continue;
			Dtype tloss1 = max(Dtype(0), dis_data[i * num + i + 1] - dis_data[i * num + j] + Dtype(margin));
			Dtype tloss2 = max(Dtype(0), dis_data[i * num + i + 1] - dis_data[(i + 1) * num + j] + Dtype(margin));
			loss += tloss1 + tloss2;
		}
	}

	loss = loss / cnt;
	top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void RankHardLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {


	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* label = bottom[1]->cpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	int count = bottom[0]->count();
	int num = bottom[0]->num();
	int dim = bottom[0]->count() / bottom[0]->num();


	RankParameter rank_param = this->layer_param_.rank_param();
	int neg_num = rank_param.neg_num();
	int pair_size = rank_param.pair_size();
	float hard_ratio = rank_param.hard_ratio();
	float rand_ratio = rank_param.rand_ratio();
	float margin = rank_param.margin();

	Dtype* dis_data = dis_.mutable_cpu_data();
	Dtype* mask_data = mask_.mutable_cpu_data();

	for(int i = 0; i < count; i ++ )
		bottom_diff[i] = 0;

	int cnt = neg_num * num / pair_size * 2;

	for(int i = 0; i < num; i += pair_size)
	{
		const Dtype* fori = bottom_data + i * dim;
	    const Dtype* fpos = bottom_data + (i + 1) * dim;

	    Dtype* fori_diff = bottom_diff + i * dim;
		Dtype* fpos_diff = bottom_diff + (i + 1) * dim;
		for(int j = 0; j < num; j ++)
		{
			if(mask_data[i * num + j] == 0) continue;
			Dtype tloss1 = max(Dtype(0), dis_data[i * num + i + 1] - dis_data[i * num + j] + Dtype(margin));
			Dtype tloss2 = max(Dtype(0), dis_data[i * num + i + 1] - dis_data[(i + 1) * num + j] + Dtype(margin));

			const Dtype* fneg = bottom_data + j * dim;
			Dtype* fneg_diff = bottom_diff + j * dim;
			if(tloss1 > 0)
			{
				for(int k = 0; k < dim; k ++)
			    {
					fori_diff[k] += (fneg[k] - fpos[k]); // / (pairNum * 1.0 - 2.0);
					fpos_diff[k] += -fori[k]; // / (pairNum * 1.0 - 2.0);
					fneg_diff[k] +=  fori[k];
			    }
			}
			if(tloss2 > 0)
			{
				for(int k = 0; k < dim; k ++)
				{
					fori_diff[k] += -fpos[k]; // / (pairNum * 1.0 - 2.0);
				    fpos_diff[k] += fneg[k]-fori[k]; // / (pairNum * 1.0 - 2.0);
				    fneg_diff[k] += fpos[k];
				}
			}

		}
	}

	for (int i = 0; i < count; i ++)
	{
		bottom_diff[i] = bottom_diff[i] / cnt;
	}

}

#ifdef CPU_ONLY
STUB_GPU(RankHardLossLayer);
#endif

INSTANTIATE_CLASS(RankHardLossLayer);
REGISTER_LAYER_CLASS(RankHardLoss);

}  // namespace caffe
