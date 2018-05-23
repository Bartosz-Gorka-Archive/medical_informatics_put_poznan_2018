import api from '@/api.js'
import constPaths from '@/constants/constant-paths.js'

const state = {
  observations: [],
  loadingObservations: true,
  nextUrl: constPaths.OBSERVATION_URL
}

const getters = {
  observations: state => state.observations,
  loadingObservations: state => state.loadingObservations
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
