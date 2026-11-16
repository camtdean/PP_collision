// SpacepointDumper: per event, the TPC clusters as TPCpp-10M stores them (E = cluster ADC, x, y, z in cm) with the
// truth of the particle that deposited most of each cluster's energy, the truth PV and sPHENIX's reconstructed
// vertices (SvtxVertexMap), into the tree "spacepoints" (one entry per event); read by spacepoints_to_dataset.py.
#ifndef SPACEPOINTDUMPER_H
#define SPACEPOINTDUMPER_H

#include <fun4all/SubsysReco.h>

#include <memory>
#include <string>
#include <vector>

class SvtxEvalStack;
class TFile;
class TTree;

class SpacepointDumper : public SubsysReco
{
 public:
  explicit SpacepointDumper(const std::string &name = "SpacepointDumper", const std::string &output = "spacepoints.root");
  ~SpacepointDumper() override;

  int InitRun(PHCompositeNode *topNode) override;
  int process_event(PHCompositeNode *topNode) override;
  int End(PHCompositeNode *topNode) override;

  void SetVertexMapName(const std::string &name) { m_vertex_map_name = name; }

 private:
  void clear();

  std::string m_output;
  std::string m_vertex_map_name = "SvtxVertexMap";
  std::unique_ptr<SvtxEvalStack> m_eval;
  TFile *m_file = nullptr;
  TTree *m_tree = nullptr;

  // event
  int m_event = 0;
  float m_truth_pv[3] = {};
  std::vector<float> m_reco_x, m_reco_y, m_reco_z, m_reco_cov;
  std::vector<int> m_reco_ntracks;
  // clusters
  std::vector<float> m_adc, m_x, m_y, m_z, m_px, m_py, m_pz, m_e, m_vtx_x, m_vtx_y, m_vtx_z, m_charge;
  std::vector<int> m_layer, m_track_id, m_parent_id, m_pid;
};

#endif
