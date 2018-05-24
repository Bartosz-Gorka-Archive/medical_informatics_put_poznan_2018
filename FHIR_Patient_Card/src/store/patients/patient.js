import api from '@/api.js'
import constPaths from '@/constants/constant-paths.js'

const state = {
  patients: [],
  loadingPatients: true,
  selectedPatient: null,
  totalVersions: 1,
  familyName: '',
  nextUrl: constPaths.PATIENT_URL + '$everything'
}

const getters = {
  patients: state => state.patients,
  loadingPatients: state => state.loadingPatients,
  familyName: state => state.familyName,
  selectedPatient: state => state.selectedPatient,
  totalVersions: state => state.totalVersions
}

const actions = {
  getPatients ({ state, commit }) {
    return api.fetch(state.nextUrl)
    .then(data => {
      commit('setList', data)
      return state.patients
    })
    .catch(error => Promise.reject(error))
  },
  getSinglePatient ({ state, commit }, patientID) {
    return api.fetch(constPaths.PATIENT_URL + patientID)
    .then(data => {
      commit('setSelectedPatient', data)
      return state.selectedPatient
    })
    .catch(error => Promise.reject(error))
  },
  getSingleVersionedPatient ({ state, commit }, { patientID, versionNumber }) {
    return api.fetch(constPaths.PATIENT_URL + patientID + '/_history/' + versionNumber)
    .then(data => {
      commit('setSelectedPatient', data)
      return state.selectedPatient
    })
    .catch(error => Promise.reject(error))
  },
  setFindByFamilyName ({ state, commit }, name) {
    commit('setFamilyNameFilter', name)
  },
  clear ({ state, commit }) {
    commit('clear')
    return state.patients
  }
}

const mutations = {
  setFamilyNameFilter (state, name) {
    state.patients = []
    state.familyName = name
    state.loadingPatients = true
    state.nextUrl = constPaths.PATIENT_URL + '?family=' + name
  },
  setSelectedPatient (state, data) {
    state.selectedPatient = data
    state.totalVersions = parseInt(data.meta.versionId)
  },
  setList (state, data) {
    state.patients = [...state.patients, ...data.entry]
    if (data.link[1] && data.link[1].relation === 'next') {
      state.nextUrl = data.link[1].url
      state.loadingPatients = true
    } else {
      state.nextUrl = constPaths.PATIENT_URL
      state.loadingPatients = false
    }
  },
  clear (state) {
    state.patients = []
    state.nextUrl = constPaths.PATIENT_URL
    state.loadingPatients = true
  }
}

export default {
  namespaced: true,
  state,
  getters,
  actions,
  mutations
}
