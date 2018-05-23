import api from '@/api.js'
import constPaths from '@/constants/constant-paths.js'

const state = {
  medications: [],
  loadingMedications: true,
  nextUrl: constPaths.MEDICATION_URL
}

const getters = {
  medications: state => state.medications,
  loadingMedications: state => state.loadingMedications
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
