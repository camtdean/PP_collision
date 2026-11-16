#include "SpacepointDumper.h"

#include <fun4all/Fun4AllReturnCodes.h>
#include <g4eval/SvtxClusterEval.h>
#include <g4eval/SvtxEvalStack.h>
#include <g4main/PHG4Particle.h>
#include <g4main/PHG4TruthInfoContainer.h>
#include <g4main/PHG4VtxPoint.h>
#include <globalvertex/SvtxVertex.h>
#include <globalvertex/SvtxVertexMap.h>
#include <phool/getClass.h>
#include <trackbase/ActsGeometry.h>
#include <trackbase/TrkrCluster.h>
#include <trackbase/TrkrClusterContainer.h>
#include <trackbase/TrkrDefs.h>

#include <TDatabasePDG.h>
#include <TFile.h>
#include <TParticlePDG.h>
#include <TTree.h>

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace
{
  // charge (e) from the PDG code; nuclei (10LZZZAAAI) from Z
  float charge_of(int pid)
  {
    if (const TParticlePDG *particle = TDatabasePDG::Instance()->GetParticle(pid))
    {
      return particle->Charge() / 3.0F;
    }
    if (std::abs(pid) > 1000000000)
    {
      return static_cast<float>((std::abs(pid) / 10000) % 1000) * (pid > 0 ? 1.0F : -1.0F);
    }
    return 0.0F;
  }
}  // namespace

SpacepointDumper::SpacepointDumper(const std::string &name, const std::string &output)
  : SubsysReco(name)
  , m_output(output)
{
}

SpacepointDumper::~SpacepointDumper() = default;

int SpacepointDumper::InitRun(PHCompositeNode * /*topNode*/)
{
  m_file = TFile::Open(m_output.c_str(), "RECREATE");
  if (!m_file || m_file->IsZombie())
  {
    std::cout << Name() << ": cannot write " << m_output << std::endl;
    return Fun4AllReturnCodes::ABORTRUN;
  }
  m_tree = new TTree("spacepoints", "TPC clusters with truth, truth PV and reconstructed vertices per event");
  m_tree->Branch("event", &m_event);
  m_tree->Branch("truth_pv_x", &m_truth_pv[0]);
  m_tree->Branch("truth_pv_y", &m_truth_pv[1]);
  m_tree->Branch("truth_pv_z", &m_truth_pv[2]);
  m_tree->Branch("reco_x", &m_reco_x);
  m_tree->Branch("reco_y", &m_reco_y);
  m_tree->Branch("reco_z", &m_reco_z);
  m_tree->Branch("reco_ntracks", &m_reco_ntracks);
  m_tree->Branch("reco_cov", &m_reco_cov);  // per vertex xx xy yy xz yz zz
  m_tree->Branch("adc", &m_adc);
  m_tree->Branch("x", &m_x);
  m_tree->Branch("y", &m_y);
  m_tree->Branch("z", &m_z);
  m_tree->Branch("layer", &m_layer);
  m_tree->Branch("track_id", &m_track_id);  // 0: no truth match
  m_tree->Branch("parent_id", &m_parent_id);
  m_tree->Branch("pid", &m_pid);
  m_tree->Branch("px", &m_px);
  m_tree->Branch("py", &m_py);
  m_tree->Branch("pz", &m_pz);
  m_tree->Branch("e", &m_e);
  m_tree->Branch("vtx_x", &m_vtx_x);
  m_tree->Branch("vtx_y", &m_vtx_y);
  m_tree->Branch("vtx_z", &m_vtx_z);
  m_tree->Branch("charge", &m_charge);
  return Fun4AllReturnCodes::EVENT_OK;
}

void SpacepointDumper::clear()
{
  for (auto *v : {&m_reco_x, &m_reco_y, &m_reco_z, &m_reco_cov, &m_adc, &m_x, &m_y, &m_z, &m_px, &m_py, &m_pz, &m_e,
                  &m_vtx_x, &m_vtx_y, &m_vtx_z, &m_charge})
  {
    v->clear();
  }
  for (auto *v : {&m_reco_ntracks, &m_layer, &m_track_id, &m_parent_id, &m_pid})
  {
    v->clear();
  }
}

int SpacepointDumper::process_event(PHCompositeNode *topNode)
{
  auto *clusters = findNode::getClass<TrkrClusterContainer>(topNode, "TRKR_CLUSTER");
  auto *geometry = findNode::getClass<ActsGeometry>(topNode, "ActsGeometry");
  auto *truth = findNode::getClass<PHG4TruthInfoContainer>(topNode, "G4TruthInfo");
  if (!clusters || !geometry || !truth)
  {
    std::cout << Name() << ": needs TRKR_CLUSTER, ActsGeometry and G4TruthInfo" << std::endl;
    return Fun4AllReturnCodes::ABORTRUN;
  }
  if (!m_eval)
  {
    m_eval = std::make_unique<SvtxEvalStack>(topNode);
  }
  m_eval->next_event(topNode);
  SvtxClusterEval *cluster_eval = m_eval->get_cluster_eval();
  clear();

  const PHG4VtxPoint *pv = truth->GetPrimaryVtx(truth->GetPrimaryVertexIndex());
  m_truth_pv[0] = pv ? pv->get_x() : NAN;
  m_truth_pv[1] = pv ? pv->get_y() : NAN;
  m_truth_pv[2] = pv ? pv->get_z() : NAN;

  if (auto *vertices = findNode::getClass<SvtxVertexMap>(topNode, m_vertex_map_name))
  {
    for (auto iter = vertices->begin(); iter != vertices->end(); ++iter)
    {
      const SvtxVertex *vertex = iter->second;
      m_reco_x.push_back(vertex->get_x());
      m_reco_y.push_back(vertex->get_y());
      m_reco_z.push_back(vertex->get_z());
      m_reco_ntracks.push_back(vertex->size_tracks());
      for (unsigned int i = 0; i < 3; ++i)
      {
        for (unsigned int j = 0; j <= i; ++j)
        {
          m_reco_cov.push_back(vertex->get_error(i, j));
        }
      }
    }
  }

  for (const auto &hitsetkey : clusters->getHitSetKeys(TrkrDefs::TrkrId::tpcId))
  {
    const auto range = clusters->getClusters(hitsetkey);
    for (auto iter = range.first; iter != range.second; ++iter)
    {
      const TrkrDefs::cluskey key = iter->first;
      TrkrCluster *cluster = iter->second;
      const auto position = geometry->getGlobalPosition(key, cluster);
      m_adc.push_back(cluster->getAdc());
      m_x.push_back(position.x());
      m_y.push_back(position.y());
      m_z.push_back(position.z());
      m_layer.push_back(TrkrDefs::getLayer(key));

      const PHG4Particle *particle = cluster_eval->max_truth_particle_by_cluster_energy(key);
      const PHG4VtxPoint *origin = particle ? truth->GetVtx(particle->get_vtx_id()) : nullptr;
      m_track_id.push_back(particle ? particle->get_track_id() : 0);
      m_parent_id.push_back(particle ? particle->get_parent_id() : 0);
      m_pid.push_back(particle ? particle->get_pid() : 0);
      m_px.push_back(particle ? particle->get_px() : NAN);
      m_py.push_back(particle ? particle->get_py() : NAN);
      m_pz.push_back(particle ? particle->get_pz() : NAN);
      m_e.push_back(particle ? particle->get_e() : NAN);
      m_charge.push_back(particle ? charge_of(particle->get_pid()) : NAN);
      m_vtx_x.push_back(origin ? origin->get_x() : NAN);
      m_vtx_y.push_back(origin ? origin->get_y() : NAN);
      m_vtx_z.push_back(origin ? origin->get_z() : NAN);
    }
  }

  m_tree->Fill();
  ++m_event;
  return Fun4AllReturnCodes::EVENT_OK;
}

int SpacepointDumper::End(PHCompositeNode * /*topNode*/)
{
  if (m_file)
  {
    m_file->cd();
    m_tree->Write();
    m_file->Close();
    delete m_file;
    m_file = nullptr;
  }
  return Fun4AllReturnCodes::EVENT_OK;
}
