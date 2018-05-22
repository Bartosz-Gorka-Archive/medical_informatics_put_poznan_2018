import api from '@/api.js'

const state = {
  patients: [],
  loadingPatients: true,
  next_url: 'http://hapi.fhir.org/baseDstu3/Patient'
}

const getters = {
  patients: state => state.patients,
  loadingPatients: state => state.loadingPatients
}

const actions = {
  getPatients ({ state, commit }) {
    return api.fetch_patients(state.next_url)
    .then(data => {
      commit('setList', data)
      return state.patients
    })
    .catch(error => Promise.reject(error))
  },
  clear ({ state, commit }) {
    commit('clear')
    return state.patients
  }
}

const mutations = {
  setList (state, data) {
    state.patients = [...state.patients, ...data.entry]
    if (data.link[1] && data.link[1].relation === 'next') {
      state.next_url = data.link[1].url
      state.loadingPatients = true
    } else {
      state.next_url = 'http://hapi.fhir.org/baseDstu3/Patient'
      state.loadingPatients = false
    }
  },
  clear (state) {
    state.patients = []
    state.next_url = 'http://hapi.fhir.org/baseDstu3/Patient'
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
