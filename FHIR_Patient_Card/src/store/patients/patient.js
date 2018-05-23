import api from '@/api.js'

const state = {
  patients: [],
  loadingPatients: true,
  familyName: '',
  nextUrl: 'http://hapi.fhir.org/baseDstu3/Patient'
}

const getters = {
  patients: state => state.patients,
  loadingPatients: state => state.loadingPatients,
  familyName: state => state.familyName
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
    state.nextUrl = 'http://hapi.fhir.org/baseDstu3/Patient?family=' + name
  },
  setList (state, data) {
    state.patients = [...state.patients, ...data.entry]
    if (data.link[1] && data.link[1].relation === 'next') {
      state.nextUrl = data.link[1].url
      state.loadingPatients = true
    } else {
      state.nextUrl = 'http://hapi.fhir.org/baseDstu3/Patient'
      state.loadingPatients = false
    }
  },
  clear (state) {
    state.patients = []
    state.nextUrl = 'http://hapi.fhir.org/baseDstu3/Patient'
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
