import api from '@/api.js'
import constPaths from '@/constants/constant-paths.js'

const state = {
  observations: [],
  selectedObservation: null,
  loadingObservations: true,
  totalVersions: 1,
  nextUrl: constPaths.OBSERVATION_URL
}

const getters = {
  observations: state => state.observations,
  loadingObservations: state => state.loadingObservations,
  totalVersions: state => state.totalVersions,
  selectedObservation: state => state.selectedObservation
}

const actions = {
  getObservations ({ state, commit }) {
    return api.fetch(state.nextUrl)
    .then(data => {
      commit('setList', data)
      return state.observations
    })
    .catch(error => Promise.reject(error))
  },
  getSingleObservation ({ state, commit }, observationID) {
    return api.fetch(constPaths.OBSERVATION_URL + observationID)
    .then(data => {
      commit('setSelectedObservation', data)
      return state.selectedObservation
    })
    .catch(error => Promise.reject(error))
  },
  getSingleVersionedObservation ({ state, commit }, { observationID, versionNumber }) {
    return api.fetch(constPaths.OBSERVATION_URL + observationID + '/_history/' + versionNumber)
    .then(data => {
      commit('setSelectedObservation', data)
      return state.selectedObservation
    })
    .catch(error => Promise.reject(error))
  },
  updateObservation ({ state, commit }, { code, observationID, display, value }) {
    // Build changes
    var content = state.selectedObservation
    content.code.text = display
    content.code.coding[0].display = display
    content.code.coding[0].code = code
    content.valueQuantity.value = value

    // Send changes
    return api.update(constPaths.OBSERVATION_URL + observationID, content)
    .then(data => {
      return api.fetch(constPaths.OBSERVATION_URL + observationID)
      .then(data => {
        commit('setSelectedObservation', data)
        return state.selectedObservation
      })
      .catch(error => Promise.reject(error))
    })
    .catch(error => Promise.reject(error))
  },
  clear ({ state, commit }) {
    commit('clear')
    return state.observations
  }
}

const mutations = {
  setList (state, data) {
    state.observations = [...state.observations, ...data.entry]
    if (data.link[1] && data.link[1].relation === 'next') {
      state.nextUrl = data.link[1].url
      state.loadingObservations = true
    } else {
      state.nextUrl = constPaths.OBSERVATION_URL
      state.loadingObservations = false
    }
  },
  setSelectedObservation (state, data) {
    state.selectedObservation = data
    state.totalVersions = parseInt(data.meta.versionId)
  },
  clear (state) {
    state.observations = []
    state.nextUrl = constPaths.OBSERVATION_URL
    state.loadingObservations = true
  }
}

export default {
  namespaced: true,
  state,
  getters,
  actions,
  mutations
}
