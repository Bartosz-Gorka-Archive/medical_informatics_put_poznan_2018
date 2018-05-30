import api from '@/api.js'
import constPaths from '@/constants/constant-paths.js'

const state = {
  patients: [],
  loadingPatients: true,
  loadingObservations: true,
  selectedPatient: null,
  observations: [],
  totalVersions: 1,
  familyName: '',
  patientDetailsURL: constPaths.PATIENT_URL,
  nextUrl: constPaths.PATIENT_URL
}

const getters = {
  patients: state => state.patients,
  loadingPatients: state => state.loadingPatients,
  loadingObservations: state => state.loadingObservations,
  familyName: state => state.familyName,
  selectedPatient: state => state.selectedPatient,
  totalVersions: state => state.totalVersions,
  observations: state => state.observations
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
  getPatientObservations ({ state, commit }) {
    return api.fetch(state.patientDetailsURL)
    .then(data => {
      commit('setObservation', data)
      return state.observations
    })
    .catch(error => Promise.reject(error))
  },
  updatePatient ({ state, commit }, { birthDate, patientID, gender }) {
    var content = state.selectedPatient
    content.birthDate = birthDate
    content.gender = gender
    return api.updatePatient(constPaths.PATIENT_URL + patientID, content)
    .then(data => {
      return api.fetch(constPaths.PATIENT_URL + patientID)
      .then(data => {
        commit('setSelectedPatient', data)
        return state.selectedPatient
      })
      .catch(error => Promise.reject(error))
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
    state.observations = []
    state.patientDetailsURL = constPaths.PATIENT_URL + data.id + '/$everything?_sort_by=date'
    state.loadingObservations = true
    state.totalVersions = parseInt(data.meta.versionId)
  },
  setObservation (state, data) {
    state.observations = [...state.observations, ...data.entry]
    state.observations.sort(function (a, b) {
      return new Date(a.resource.meta.lastUpdated).getTime() <= new Date(b.resource.meta.lastUpdated).getTime()
    })
    if (data.link[1] && data.link[1].relation === 'next') {
      state.patientDetailsURL = data.link[1].url
      state.loadingObservations = true
    } else {
      state.loadingObservations = false
    }
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
