import api from '@/api.js'
import constPaths from '@/constants/constant-paths.js'

const state = {
  medications: [],
  selectedMedication: null,
  loadingMedications: true,
  totalVersions: 1,
  nextUrl: constPaths.MEDICATION_URL
}

const getters = {
  medications: state => state.medications,
  loadingMedications: state => state.loadingMedications,
  totalVersions: state => state.totalVersions,
  selectedMedication: state => state.selectedMedication
}

const actions = {
  getMedications ({ state, commit }) {
    return api.fetch(state.nextUrl)
    .then(data => {
      commit('setList', data)
      return state.medications
    })
    .catch(error => Promise.reject(error))
  },
  getSingleMedication ({ state, commit }, medicationID) {
    return api.fetch(constPaths.MEDICATION_URL + medicationID)
    .then(data => {
      commit('setSelectedMedication', data)
      return state.selectedMedication
    })
    .catch(error => Promise.reject(error))
  },
  getSingleVersionedMedication ({ state, commit }, { medicationID, versionNumber }) {
    return api.fetch(constPaths.MEDICATION_URL + medicationID + '/_history/' + versionNumber)
    .then(data => {
      commit('setSelectedMedication', data)
      return state.selectedMedication
    })
    .catch(error => Promise.reject(error))
  },
  clear ({ state, commit }) {
    commit('clear')
    return state.medications
  }
}

const mutations = {
  setList (state, data) {
    state.medications = [...state.medications, ...data.entry]
    if (data.link[1] && data.link[1].relation === 'next') {
      state.nextUrl = data.link[1].url
      state.loadingMedications = true
    } else {
      state.nextUrl = constPaths.MEDICATION_URL
      state.loadingMedications = false
    }
  },
  setSelectedMedication (state, data) {
    state.selectedMedication = data
    state.totalVersions = parseInt(data.meta.versionId)
  },
  clear (state) {
    state.medications = []
    state.nextUrl = constPaths.MEDICATION_URL
    state.loadingMedications = true
  }
}

export default {
  namespaced: true,
  state,
  getters,
  actions,
  mutations
}
